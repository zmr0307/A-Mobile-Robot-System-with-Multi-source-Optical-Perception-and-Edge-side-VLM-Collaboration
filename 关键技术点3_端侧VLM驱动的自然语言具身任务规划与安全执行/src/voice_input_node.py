#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WL100 语音输入节点 (Voice Input)

职责：
  把用户的麦克风语音 → 中文/多语言文本，通过话题暴露给 panel。
  使用 sherpa-onnx + SenseVoice INT8 离线推理，全本地、零网络。

亮点（写在 system 里方便后续维护）：
  ✓ SenseVoice-small INT8 (229MB)，CER 4% 中英日韩粤五语言通吃
  ✓ 含语言识别 / 情感识别 / 事件检测（笑声/掌声）/ 标点恢复 / 逆文本归一化
  ✓ 实时音量 RMS 推送，给 panel 画动态音量条
  ✓ 推理 0.5s 完成 5 秒中文音频（Jetson Orin NX 16GB）
  ✓ NoMachine 客户端麦克风转发（pulseaudio source: nx_remapped_out）开箱即用

订阅：
  /demo/voice/control  (std_msgs/String) data="start" 开录 / data="stop" 收尾识别

发布：
  /demo/voice/state       (std_msgs/String) JSON 状态流，给 panel 显示
        idle      → {"state":"idle"}
        recording → {"state":"recording","duration_sec":1.5,"volume_db":-20.5,
                     "volume_pct":0.55}
        thinking  → {"state":"thinking","duration_sec":2.3}
        transcribed → {"state":"transcribed","text":"...","lang":"zh",
                       "emotion":"NEUTRAL","event":"Speech",
                       "audio_sec":2.3,"infer_sec":0.5}
        error     → {"state":"error","message":"..."}
  /demo/voice/transcript  (std_msgs/String) 仅识别文本，便于直接订阅消费
"""

import json
import os
import threading
import time
from collections import deque
from typing import Optional

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

try:
    import sounddevice as sd
except Exception as e:
    sd = None
    _SD_IMPORT_ERR = e
else:
    _SD_IMPORT_ERR = None

try:
    import sherpa_onnx
except Exception as e:
    sherpa_onnx = None
    _SH_IMPORT_ERR = e
else:
    _SH_IMPORT_ERR = None


DEFAULT_MODEL_DIR = (
    "/home/nvidia/asr_models/"
    "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17"
)


class VoiceInputNode(Node):
    def __init__(self):
        super().__init__("voice_input")

        # ───── 参数 ─────
        self.declare_parameter("model_dir", DEFAULT_MODEL_DIR)
        self.declare_parameter("model_file", "model.int8.onnx")
        self.declare_parameter("tokens_file", "tokens.txt")
        self.declare_parameter("language", "auto")  # auto/zh/en/ja/ko/yue
        self.declare_parameter("use_itn", True)     # 逆文本归一化
        self.declare_parameter("num_threads", 4)
        self.declare_parameter("sample_rate", 16000)
        self.declare_parameter("input_device", "")  # "" → pulseaudio default
        self.declare_parameter("max_record_sec", 30.0)
        self.declare_parameter("min_record_sec", 0.3)
        self.declare_parameter("rms_window_sec", 0.1)
        self.declare_parameter("control_topic", "/demo/voice/control")
        self.declare_parameter("state_topic", "/demo/voice/state")
        self.declare_parameter("transcript_topic", "/demo/voice/transcript")

        self.model_dir = str(self.get_parameter("model_dir").value)
        self.model_file = str(self.get_parameter("model_file").value)
        self.tokens_file = str(self.get_parameter("tokens_file").value)
        self.language = str(self.get_parameter("language").value)
        self.use_itn = bool(self.get_parameter("use_itn").value)
        self.num_threads = int(self.get_parameter("num_threads").value)
        self.sample_rate = int(self.get_parameter("sample_rate").value)
        self.input_device = str(self.get_parameter("input_device").value)
        self.max_record_sec = float(
            self.get_parameter("max_record_sec").value)
        self.min_record_sec = float(
            self.get_parameter("min_record_sec").value)
        self.rms_window_sec = float(
            self.get_parameter("rms_window_sec").value)

        ctrl_topic = str(self.get_parameter("control_topic").value)
        state_topic = str(self.get_parameter("state_topic").value)
        trans_topic = str(self.get_parameter("transcript_topic").value)

        # ───── 依赖检查 ─────
        if sd is None:
            self.get_logger().fatal(
                f"sounddevice 不可用: {_SD_IMPORT_ERR}。"
                f"请先 pip install sounddevice + apt portaudio19-dev。")
            raise SystemExit(1)
        if sherpa_onnx is None:
            self.get_logger().fatal(
                f"sherpa-onnx 不可用: {_SH_IMPORT_ERR}。"
                f"请先 pip install sherpa-onnx。")
            raise SystemExit(1)

        # ───── 模型加载（一次性，启动慢 3-4 秒）─────
        model_path = os.path.join(self.model_dir, self.model_file)
        tokens_path = os.path.join(self.model_dir, self.tokens_file)
        if not os.path.isfile(model_path) or not os.path.isfile(tokens_path):
            self.get_logger().fatal(
                f"模型文件缺失：\n  model = {model_path}\n  tokens = {tokens_path}")
            raise SystemExit(1)

        self.get_logger().info(f"⏳ 加载 SenseVoice INT8 模型 ...")
        t0 = time.time()
        self.recognizer = sherpa_onnx.OfflineRecognizer.from_sense_voice(
            model=model_path,
            tokens=tokens_path,
            num_threads=self.num_threads,
            language=self.language,
            use_itn=self.use_itn,
            debug=False,
        )
        self.get_logger().info(f"✅ 模型加载完成 ({time.time() - t0:.1f}s)")

        # ───── 录音状态 ─────
        self.rec_lock = threading.Lock()
        self.recording = False
        self.audio_chunks: list = []  # list of float32 numpy arrays
        self.record_start_t = 0.0
        self.input_stream: Optional[sd.InputStream] = None
        # 最近一段窗口的样本（用于 RMS 计算）
        self.rms_buffer = deque(
            maxlen=int(self.sample_rate * self.rms_window_sec))

        # 推理锁（防止 panel 短时间内连发多个 stop）
        self.infer_busy_lock = threading.Lock()
        self.infer_busy = False

        # ───── ROS 接口 ─────
        self.create_subscription(
            String, ctrl_topic, self._on_control, 10)
        self.state_pub = self.create_publisher(String, state_topic, 10)
        self.transcript_pub = self.create_publisher(String, trans_topic, 10)

        # 周期推送 state（让 panel 拿到 RMS / 时长更新）
        self.create_timer(0.1, self._tick_state)

        # 启动横幅
        self.get_logger().info("════════════════════════════════════════")
        self.get_logger().info("🎙️  WL100 语音输入节点已启动")
        self.get_logger().info(f"  模型路径:   {model_path}")
        self.get_logger().info(f"  语言:       {self.language}  (ITN={self.use_itn})")
        self.get_logger().info(f"  采样率:     {self.sample_rate} Hz")
        self.get_logger().info(
            f"  输入设备:   {self.input_device or 'pulseaudio default'}")
        self.get_logger().info(f"  控制话题:   {ctrl_topic}")
        self.get_logger().info(f"  状态话题:   {state_topic}")
        self.get_logger().info(f"  文本话题:   {trans_topic}")
        self.get_logger().info(f"  最长录音:   {self.max_record_sec:.0f}s")
        self.get_logger().info("════════════════════════════════════════")
        self.get_logger().info("💡 测试: ros2 topic pub --once /demo/voice/control "
                                "std_msgs/msg/String \"{data: 'start'}\"")
        self._publish_state({"state": "idle"})

    # ════════════════════════════════════════════════
    #  控制入口
    # ════════════════════════════════════════════════
    def _on_control(self, msg: String):
        cmd = (msg.data or "").strip().lower()
        if cmd == "start":
            self._start_record()
        elif cmd == "stop":
            self._stop_and_transcribe()
        elif cmd == "cancel":
            self._cancel_record()
        else:
            self.get_logger().warn(f"未知 control 命令: {cmd!r}")

    # ════════════════════════════════════════════════
    #  录音
    # ════════════════════════════════════════════════
    def _audio_cb(self, indata, frames, time_info, status):
        if status:
            self.get_logger().warn(f"sounddevice 状态: {status}")
        # indata: (frames, channels) float32
        # SenseVoice 要 16kHz mono → input_stream 已经设好就是这格式
        chunk = indata[:, 0].copy() if indata.ndim == 2 else indata.copy()
        with self.rec_lock:
            if self.recording:
                self.audio_chunks.append(chunk)
                # 把最后一段填进 rms 窗
                self.rms_buffer.extend(chunk.tolist())
        # 超时保护：超过最长录音自动停
        if (time.time() - self.record_start_t) > self.max_record_sec:
            self.get_logger().warn(
                f"⏱ 录音达到最长上限 {self.max_record_sec:.0f}s，自动 stop")
            threading.Thread(
                target=self._stop_and_transcribe, daemon=True).start()

    def _start_record(self):
        with self.rec_lock:
            if self.recording:
                self.get_logger().warn("已在录音中，忽略重复 start")
                return
            self.audio_chunks = []
            self.rms_buffer.clear()
            self.recording = True
            self.record_start_t = time.time()
        try:
            device = self.input_device if self.input_device else None
            self.input_stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype="float32",
                blocksize=int(self.sample_rate * 0.05),  # 50ms 块
                callback=self._audio_cb,
                device=device,
            )
            self.input_stream.start()
            self.get_logger().info("🔴 开始录音 ...")
        except Exception as e:
            with self.rec_lock:
                self.recording = False
            self.get_logger().error(f"录音启动失败: {e}")
            self._publish_state({"state": "error",
                                 "message": f"录音启动失败: {e}"})

    def _stop_record_stream(self):
        try:
            if self.input_stream is not None:
                self.input_stream.stop()
                self.input_stream.close()
        except Exception as e:
            self.get_logger().warn(f"关闭流异常: {e}")
        finally:
            self.input_stream = None

    def _cancel_record(self):
        with self.rec_lock:
            if not self.recording:
                return
            self.recording = False
        self._stop_record_stream()
        self.audio_chunks = []
        self.rms_buffer.clear()
        self.get_logger().info("🛑 录音已取消")
        self._publish_state({"state": "idle"})

    def _stop_and_transcribe(self):
        with self.rec_lock:
            if not self.recording:
                self.get_logger().warn("当前不在录音，忽略 stop")
                return
            self.recording = False
            chunks = self.audio_chunks
            self.audio_chunks = []
            duration = time.time() - self.record_start_t
        self._stop_record_stream()

        if duration < self.min_record_sec or not chunks:
            self.get_logger().warn(
                f"录音过短 {duration:.2f}s（< {self.min_record_sec}s），跳过推理")
            self._publish_state({
                "state": "error",
                "message": f"录音过短（{duration:.1f}s），请按住说一句完整的话",
            })
            return

        with self.infer_busy_lock:
            if self.infer_busy:
                self.get_logger().warn("上一次推理还没完成，跳过")
                return
            self.infer_busy = True

        threading.Thread(
            target=self._do_transcribe,
            args=(chunks, duration), daemon=True).start()

    # ════════════════════════════════════════════════
    #  推理
    # ════════════════════════════════════════════════
    def _do_transcribe(self, chunks, audio_duration):
        try:
            audio = np.concatenate(chunks).astype(np.float32)
            self._publish_state({
                "state": "thinking",
                "duration_sec": float(audio_duration),
            })
            self.get_logger().info(
                f"🧠 推理中 ... 音频 {audio_duration:.2f}s, "
                f"{len(audio)} samples")

            t0 = time.time()
            stream = self.recognizer.create_stream()
            stream.accept_waveform(self.sample_rate, audio)
            self.recognizer.decode_stream(stream)
            infer_sec = time.time() - t0

            r = stream.result
            text = (r.text or "").strip()
            # SenseVoice 的 lang/emotion/event 是 "<|zh|>" 这种格式，剥一下
            lang = self._strip_token(getattr(r, "lang", ""))
            emotion = self._strip_token(getattr(r, "emotion", ""))
            event = self._strip_token(getattr(r, "event", ""))

            self.get_logger().info(
                f"✅ 识别完成 ({infer_sec:.2f}s, {audio_duration:.1f}s 音频): "
                f"{text!r} [lang={lang}, emo={emotion}, event={event}]")

            # 发文本
            tm = String()
            tm.data = text
            self.transcript_pub.publish(tm)

            # 发完整状态
            self._publish_state({
                "state": "transcribed",
                "text": text,
                "lang": lang,
                "emotion": emotion,
                "event": event,
                "audio_sec": round(audio_duration, 2),
                "infer_sec": round(infer_sec, 2),
            })
        except Exception as e:
            self.get_logger().error(f"推理失败: {e}")
            import traceback
            self.get_logger().error(traceback.format_exc())
            self._publish_state({
                "state": "error",
                "message": f"推理失败: {e}",
            })
        finally:
            with self.infer_busy_lock:
                self.infer_busy = False

    @staticmethod
    def _strip_token(s: str) -> str:
        """'<|zh|>' → 'zh'，空 → 空"""
        if not s:
            return ""
        s = s.strip()
        if s.startswith("<|") and s.endswith("|>"):
            return s[2:-2]
        return s

    # ════════════════════════════════════════════════
    #  状态推送（10 Hz）
    # ════════════════════════════════════════════════
    def _tick_state(self):
        with self.rec_lock:
            if not self.recording:
                return
            duration = time.time() - self.record_start_t
            buf = list(self.rms_buffer)
        if not buf:
            volume_db = -80.0
            volume_pct = 0.0
        else:
            arr = np.asarray(buf, dtype=np.float32)
            rms = float(np.sqrt(np.mean(arr * arr) + 1e-12))
            # 转 dB（参考满刻度 1.0），rms=1.0 → 0dB；rms=0.001 → -60dB
            volume_db = 20.0 * np.log10(rms + 1e-9)
            # 把 -60..0 dB 映射到 0..1（panel 进度条用）
            volume_pct = float(np.clip((volume_db + 60.0) / 60.0, 0.0, 1.0))
            volume_db = float(volume_db)

        self._publish_state({
            "state": "recording",
            "duration_sec": round(duration, 2),
            "volume_db": round(volume_db, 1),
            "volume_pct": round(volume_pct, 3),
        })

    # ════════════════════════════════════════════════
    #  发布工具
    # ════════════════════════════════════════════════
    def _publish_state(self, obj: dict):
        msg = String()
        msg.data = json.dumps(obj, ensure_ascii=False)
        self.state_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    try:
        node = VoiceInputNode()
    except SystemExit:
        rclpy.shutdown()
        return
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
