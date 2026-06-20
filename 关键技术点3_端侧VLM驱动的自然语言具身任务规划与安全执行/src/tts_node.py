#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WL100 语音播报节点 (TTS)

职责：
  订阅系统关键话题，把文本转语音播出来。
  使用 edge-tts（微软免费 API）+ ffplay 播放。

播报来源：
  /demo/nl_reply             → nl_parser 的回复（命令确认 / 闲聊 / 历史查询）
  /demo/director_status      → 关键事件（approaching / handover / no_person 等）
  /demo/narration_result     → VLM 解说的 description
  /demo/tts/done             → 每条语音完成/失败/打断通知

架构：
  多源文本 → 入 FIFO 队列 → 单线程串行：edge-tts 合成 mp3 → ffplay 播放
  取消时 kill ffplay + 清队列
"""

import json
import os
import subprocess
import tempfile
import threading
import time
from collections import deque

import rclpy
from rclpy.node import Node
from std_msgs.msg import Empty, String


class TtsNode(Node):
    def __init__(self):
        super().__init__("tts")

        # 参数
        self.declare_parameter("voice", "zh-CN-XiaoyiNeural")
        self.declare_parameter("rate", "+10%")
        self.declare_parameter("volume", "+0%")
        self.declare_parameter("max_queue", 5)
        self.declare_parameter("play_command", "ffplay -nodisp -autoexit")
        self.declare_parameter("tts_timeout", 60.0)
        self.declare_parameter("nl_reply_topic", "/demo/nl_reply")
        self.declare_parameter("director_status_topic",
                               "/demo/director_status")
        self.declare_parameter("narration_result_topic",
                               "/demo/narration_result")
        self.declare_parameter("cancel_topic", "/demo/cancel")
        self.declare_parameter("done_topic", "/demo/tts/done")
        self.declare_parameter("broadcast_steps", [
            "approaching",
            "wait_handover", "handover_ok",
            "handover_timeout", "handover_settle", "handover_skip",
            "no_person_skip", "rotate_search"])
        self.declare_parameter("announce_home_done", True)
        self.declare_parameter("max_text_len", 500)

        self.voice = self.get_parameter("voice").value
        self.rate = self.get_parameter("rate").value
        self.volume = self.get_parameter("volume").value
        self.max_queue = int(self.get_parameter("max_queue").value)
        self.play_cmd = self.get_parameter("play_command").value
        self.tts_timeout = float(self.get_parameter("tts_timeout").value)
        nl_topic = self.get_parameter("nl_reply_topic").value
        dir_topic = self.get_parameter("director_status_topic").value
        narr_topic = self.get_parameter("narration_result_topic").value
        cancel_topic = self.get_parameter("cancel_topic").value
        done_topic = self.get_parameter("done_topic").value
        self.broadcast_steps = set(
            self.get_parameter("broadcast_steps").value)
        self.announce_home_done = bool(
            self.get_parameter("announce_home_done").value)
        self.max_text_len = int(self.get_parameter("max_text_len").value)

        # 播放队列 + 工作线程
        self.queue = deque()
        self.queue_lock = threading.Lock()
        self.queue_event = threading.Event()
        self.shutdown_flag = False
        self.current_process = None  # ffplay 子进程
        self.current_item = None
        self.interrupt_generation = 0
        self.process_lock = threading.Lock()

        self.worker_thread = threading.Thread(
            target=self._worker, daemon=True)
        self.worker_thread.start()

        # 订阅
        self.create_subscription(
            String, nl_topic, self._on_nl_reply, 10)
        self.create_subscription(
            String, dir_topic, self._on_director_status, 10)
        self.create_subscription(
            String, narr_topic, self._on_narration_result, 10)
        self.create_subscription(
            Empty, cancel_topic, self._on_cancel, 10)
        self.done_pub = self.create_publisher(String, done_topic, 10)

        self.get_logger().info("════════════════════════════════════════")
        self.get_logger().info("🔊 TTS 语音播报节点已启动")
        self.get_logger().info(f"  声音:    {self.voice}")
        self.get_logger().info(f"  语速:    {self.rate}")
        self.get_logger().info(f"  队列:    max {self.max_queue}")
        self.get_logger().info(f"  播放:    {self.play_cmd}")
        self.get_logger().info(f"  完成话题: {done_topic}")
        self.get_logger().info(f"  播报事件: {self.broadcast_steps}")
        self.get_logger().info("════════════════════════════════════════")

    # ════════════════════════════════════════════════
    #  话题回调
    # ════════════════════════════════════════════════
    def _on_nl_reply(self, msg: String):
        """nl_parser 的回复 → 全播"""
        text = (msg.data or "").strip()
        if not text:
            return
        # 跳过纯 emoji / 太短的
        if len(text) < 2:
            return
        self._enqueue(text, source="nl_reply", raw_text=text)

    def _on_director_status(self, msg: String):
        """director 关键事件 → 播 message 字段"""
        try:
            obj = json.loads(msg.data) if msg.data else {}
        except json.JSONDecodeError:
            return
        step = str(obj.get("step", "")).strip()
        message = str(obj.get("message", "")).strip()
        stage = str(obj.get("stage", "")).strip()

        # 归位完成特殊播报
        if (self.announce_home_done and step == "end"
                and stage == "home"):
            self._enqueue(
                "全部完成，已回到归位点。",
                source="director_status",
                raw_text=message,
                speech_id=str(obj.get("speech_id", "")).strip(),
                meta={"step": step, "stage": stage})
            return

        # 只播配置里列的关键 step
        if step in self.broadcast_steps and message:
            # 去掉 emoji 前缀（message 里带了 📦📍🔄 等）
            clean = message.lstrip(
                "📦📍🔄⚠️✅⏳🎙️🚀🛑🟢🧭❌ ")
            if clean:
                self._enqueue(
                    clean,
                    source="director_status",
                    raw_text=message,
                    speech_id=str(obj.get("speech_id", "")).strip(),
                    meta={"step": step, "stage": stage})

    def _on_narration_result(self, msg: String):
        """VLM 解说 → 播 description"""
        try:
            obj = json.loads(msg.data) if msg.data else {}
        except json.JSONDecodeError:
            return
        # 只播真正的 VLM 解说（有 trigger.mode），不播系统事件
        trigger = obj.get("trigger", {}) or {}
        mode = trigger.get("mode", "")
        # 系统事件（rotate_search/wait_handover/handover_ok 等）
        # 已经在 _on_director_status 里播了，这里跳过
        system_modes = {"rotate_search", "wait_handover", "handover_ok",
                        "handover_timeout", "handover_settle",
                        "handover_skip", "no_person_skip",
                        "semantic_search", "semantic_verify",
                        "semantic_matched", "semantic_not_found",
                        "semantic_candidate_rejected"}
        if mode in system_modes:
            return

        desc = str(obj.get("description", "")).strip()
        if not desc:
            return
        trigger = obj.get("trigger", {}) or {}
        self._enqueue(
            desc,
            source="narration_result",
            raw_text=desc,
            speech_id=str(obj.get("speech_id", "")).strip(),
            meta={
                "mode": str(trigger.get("mode", "")).strip(),
                "waypoint": str(trigger.get("waypoint", "")).strip(),
            })

    def _on_cancel(self, _msg):
        """取消 → 立刻停播 + 清队列"""
        self._interrupt()

    # ════════════════════════════════════════════════
    #  队列管理
    # ════════════════════════════════════════════════
    def _enqueue(self, text: str, source: str = "", raw_text: str = "",
                 speech_id: str = "", meta=None):
        # 截断过长文本
        raw_text = raw_text or text
        if len(text) > self.max_text_len:
            text = text[:self.max_text_len] + "。"
        item = {
            "text": text,
            "raw_text": raw_text,
            "source": source,
            "speech_id": speech_id or "",
            "meta": meta or {},
        }
        with self.queue_lock:
            if len(self.queue) >= self.max_queue:
                dropped = self.queue.popleft()
                self._publish_done(dropped, "dropped", "queue_full")
            self.queue.append(item)
        self.queue_event.set()
        self.get_logger().info(f"🔊 入队: {text[:40]}...")

    def _interrupt(self):
        """打断当前播放 + 清空队列"""
        interrupted = []
        with self.queue_lock:
            interrupted.extend(list(self.queue))
            self.queue.clear()
        with self.process_lock:
            self.interrupt_generation += 1
            if self.current_item:
                interrupted.append(self.current_item)
            if self.current_process and self.current_process.poll() is None:
                try:
                    self.current_process.kill()
                except Exception:
                    pass
        for item in interrupted:
            self._publish_done(item, "interrupted", "cancel")
        self.get_logger().info("🔇 播报已打断")

    def _publish_done(self, item, status: str, detail: str = ""):
        msg = String()
        msg.data = json.dumps({
            "status": status,
            "detail": detail,
            "source": item.get("source", ""),
            "speech_id": item.get("speech_id", ""),
            "text": item.get("text", ""),
            "raw_text": item.get("raw_text", ""),
            "meta": item.get("meta", {}),
            "ts": time.time(),
        }, ensure_ascii=False)
        self.done_pub.publish(msg)

    # ════════════════════════════════════════════════
    #  工作线程：串行合成 + 播放
    # ════════════════════════════════════════════════
    def _worker(self):
        while not self.shutdown_flag:
            self.queue_event.wait(timeout=1.0)
            self.queue_event.clear()
            while True:
                with self.queue_lock:
                    if not self.queue:
                        break
                    item = self.queue.popleft()
                self._synthesize_and_play(item)

    def _synthesize_and_play(self, item):
        """edge-tts 合成 → ffplay 播放 → 删临时文件"""
        text = item.get("text", "")
        tmp_path = None
        with self.process_lock:
            start_generation = self.interrupt_generation
            self.current_item = item
        status = "done"
        detail = ""
        try:
            # 1. 合成
            tmp_fd, tmp_path = tempfile.mkstemp(suffix=".mp3")
            os.close(tmp_fd)

            cmd = [
                "edge-tts",
                "--voice", self.voice,
                "--rate", self.rate,
                "--volume", self.volume,
                "--text", text,
                "--write-media", tmp_path,
            ]
            result = subprocess.run(
                cmd, capture_output=True, timeout=self.tts_timeout)
            if result.returncode != 0:
                status = "failed"
                detail = "edge_tts_failed"
                self.get_logger().warn(
                    f"edge-tts 失败: {result.stderr.decode()[:200]}")
                return

            if not os.path.isfile(tmp_path) or os.path.getsize(tmp_path) < 100:
                status = "failed"
                detail = "empty_media"
                self.get_logger().warn("edge-tts 输出文件异常")
                return

            with self.process_lock:
                if self.interrupt_generation != start_generation:
                    return

            # 2. 播放
            play_parts = self.play_cmd.split() + [tmp_path]
            with self.process_lock:
                self.current_process = subprocess.Popen(
                    play_parts,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL)
            self.current_process.wait()
            with self.process_lock:
                self.current_process = None

        except subprocess.TimeoutExpired:
            status = "failed"
            detail = "edge_tts_timeout"
            self.get_logger().warn("edge-tts 合成超时（无网络？）")
        except Exception as e:
            status = "failed"
            detail = str(e)
            self.get_logger().warn(f"TTS 异常: {e}")
        finally:
            should_publish = False
            with self.process_lock:
                if self.interrupt_generation == start_generation:
                    should_publish = True
                if self.current_item is item:
                    self.current_item = None
                self.current_process = None
            if should_publish:
                self._publish_done(item, status, detail)
            if tmp_path and os.path.isfile(tmp_path):
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass


def main(args=None):
    rclpy.init(args=args)
    node = TtsNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown_flag = True
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
