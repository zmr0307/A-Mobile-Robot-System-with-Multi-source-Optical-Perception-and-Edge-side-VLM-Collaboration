#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WL100 场景解说节点 (Scene Narrator)

数据流：
  /demo/narrate_request (上层触发，JSON 字符串)
    ↓
  scene_narrator_node：
    1. 抓最新一帧 RGB → JPEG 关键帧落盘
    2. 收集近 1 秒内的 YOLO 检测（类别、置信度、深度距离）
    3. 拼装 prompt → POST llama-server → 解析 JSON
    4. 解析失败时走格式保护处理
    ↓
  /demo/narration_result (输出，JSON 字符串)

请求格式 (/demo/narrate_request):
  {"mode": "arrive", "waypoint": "A"}
  支持 mode: arrive / find_person / no_person_advice / patrol

输出格式 (/demo/narration_result):
  {
    "trigger": {"mode": "arrive", "waypoint": "A"},
    "description": "我已到达 A 观测点...",
    "has_person": false,
    "person_count": 0,
    "person_position_hint": null,
    "key_objects": ["椅子", "纸箱"],
    "advice": null,
    "image_path": "/home/.../logs/narrate_20260526_043500_arrive_A.jpg",
    "duration_sec": 2.34,
    "fallback": false
  }

设计原则：
  - 完全独立节点，不修改 wl100_vlm 等保护包
  - 与 vlm_node 共享 llama-server 后端（端口 8080）
  - 推理排队：同时只跑 1 个推理，新请求若处于推理中则忽略
"""

import base64
import json
import os
import re
import time
import threading
from collections import deque
from datetime import datetime
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import (
    qos_profile_sensor_data,
    QoSProfile, QoSReliabilityPolicy,
    QoSDurabilityPolicy, QoSHistoryPolicy)
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import String, Empty
from vision_msgs.msg import Detection2DArray


# ════════════════════════════════════════════════════════
#  YOLO 英文类名 → 中文映射（与 semantic_nav_node 共享语义）
# ════════════════════════════════════════════════════════
EN_TO_ZH = {
    "person": "人",
    "chair": "椅子",
    "table": "桌子",
    "shelf": "货架",
    "door": "门",
    "cardboard box": "纸箱",
    "trash can": "垃圾桶",
    "robot": "机器人",
    "potted plant": "盆栽",
    "blackboard": "白板",
    "cable": "线缆",
    "power strip": "插排",
    "electric cord": "电线",
    "plug": "插头",
}


def _parse_detection_distance(det_id: str):
    """从 yolo_trt_node 的 det.id 解析距离，格式如 'person 1.42m'。"""
    try:
        parts = str(det_id or "").strip().split()
        if len(parts) < 2 or not parts[-1].endswith("m"):
            return None
        value = parts[-1][:-1]
        if value == "--":
            return None
        distance_m = float(value)
        if distance_m <= 0.0:
            return None
        return distance_m
    except (TypeError, ValueError):
        return None


# ════════════════════════════════════════════════════════
#                      System Prompt
# ════════════════════════════════════════════════════════
SYSTEM_PROMPT = """你是办公室巡检移动机器人 WL100 的场景解说员。

你接收：
  1. 当前帧画面（一张图）
  2. 本轮文字上下文：用户原始请求（如有）、waypoint、mode、视觉辅助检测。
     如有用户原始请求，需要认真参考，用来把握描述重点、篇幅和语气。
     视觉辅助检测可能包含类别、置信度和深度距离，可作为参考线索。
     如果图片中能确认该目标，并且距离与画面空间关系合理，可以在
     description 里自然表述"约 X 米处有..."（yolo观测并不一定准确！！！）。
     如果视觉辅助检测或深度距离与图片观察不一致，以实际画面描述为准。
     如果用户要求更长描述，只能扩展当前图片中能确认或不能确认的内容，不要编造。
     画面里你看不到或不能确认的，即便检测报了，也不要写进结论。
  3.最好描述一下空间的结构！！！！

你只输出一个 JSON 对象，不要有任何其他文字、代码块、推理过程：
{
  "description": "<中文描述>",
  "has_person": true | false,
  "person_count": <整数 ≥ 0>,
  "person_position_hint": "<中文方位>" | null,
  "key_objects": ["<物体>", ...],
  "advice": "<中文建议>" | null
}

【模式与字段长度规则】
  每一次场景解说的 description 都必须大于 100 个汉字。
  如果用户原始请求明确要求更长篇幅，description 按用户要求继续展开；
  但只能扩展当前图片中能确认或不能确认的内容，不要为了凑字编造。
  用户没有明确长度要求时，按下面的默认长度：
  arrive            到达航点常规解说，description 约 120-180 字，必填 key_objects
  find_person       找到人后近距离描述，description 约 110-160 字
                    ▸ 画面中确实有人 → 描述这个人的特征和位置
                    ▸ 画面中实际没人（YOLO 误报或画面已变） → has_person=false，
                      description 需要说明已查看的画面范围、未确认到人员的依据，
                      不要描述无关物体清单，key_objects 留空数组
  no_person_advice  没找到人后给出建议，description 必须大于 100 字，建议 110-150 字，必填 advice
  patrol            巡检模式观测点描述，description 约 120-180 字，必填 key_objects

【字段填写规则】

description：
  arrive          - 详细描述位置 + 看到的物体 + 有没有人。
                    例: "我已到达 A 观测点，桌面整洁，旁边放着一个白色纸箱和两把椅子，
                         没有看到工作人员。"
  find_person     - 有人时：描述这个人的特征和位置。
                    如果 YOLO 深度距离与画面判断一致，可以自然加入约略距离。
                    例（有人）: "前方约一米半处站着一个人，正面对我。"
                    无人时：明确说明未找到，并说明当前视角能确认的搜索范围和不确定区域。
                    例（无人）: "当前视野中暂未发现人员。"
  no_person_advice - description 不能只写一句话，必须大于 100 字。
                    需要包含：未发现人员结论、已查看的主要可见区域、
                    没有确认到哪些人员特征、可能存在的遮挡/盲区、下一步建议。
                    advice 字段再单独给出一句简短建议。
                    例: "本航点未发现人员，当前视角已覆盖主要通道和桌面附近区域。"
  patrol           - 平铺直叙描述观测点。
                    例: "B 观测点空着，桌上有两个纸箱，地上有一根线缆。"

has_person（保守判断）：
  true 的条件：明显看到完整人形或清晰上半身。
  以下一律填 false：
    - 只有桌椅、纸箱、空地
    - 远处模糊小色块或人形阴影
    - 仅看到衣物 / 头发碎片
    - YOLO 报告了 person 但画面中你无法亲自确认完整人形
  注：YOLO 列表里没有"人"时也要看图判断；只有图中能明确确认人形时才填 true。

person_count：
  has_person=false → 0
  has_person=true  → 数清楚画面中明显的人数，看不清估 1。

person_position_hint：
  has_person=true → 中文方位（不要数字距离）
    例: "通道中央"、"画面左侧"、"靠近右墙"、"观测点前方"
  has_person=false → null

key_objects：
  arrive / patrol 模式必须列 1~5 个最显眼的办公物体（中文）。
  其他模式可以列空数组 []。
  优先词汇：椅子、桌子、货架、门、纸箱、垃圾桶、机器人、盆栽、白板、线缆、插排、电线、插头。
  不要列墙壁、地面、天花板等环境元素。
  即使画面中有人，"人"也不要进 key_objects（已用 has_person 表达）。

advice：
  仅 mode=no_person_advice 时填中文建议，其他模式固定填 null。
  推荐建议：
    - "建议前往下一个航点继续搜索"
    - "建议返回归位点等待下一条指令"

【输出示例】

示例 1 — mode=arrive  waypoint=A
{
  "description": "我已到达 A 观测点，当前画面里能看到一张办公桌、几把椅子和一个白色纸箱，桌面整体比较整洁，通道区域也没有明显杂物阻挡。画面正前方和左右两侧都没有看到清晰的人形或工作人员，上半身、头部轮廓等人员特征也没有被确认。可见区域以办公桌和纸箱附近为主，远处遮挡部分无法进一步判断。",
  "has_person": false,
  "person_count": 0,
  "person_position_hint": null,
  "key_objects": ["椅子", "纸箱", "桌子"],
  "advice": null
}

示例 2 — mode=arrive  waypoint=B
{
  "description": "我已到达 B 观测点，当前画面中办公桌前可以确认有一位工作人员，人体轮廓和上半身都比较清楚，位置在观测点前方靠近桌面一侧。周围还能看到椅子、桌子和一些办公物品，通道没有明显拥堵。人员附近没有看到明显危险物或大面积遮挡，整体环境比较整洁，适合继续执行找人或交接相关任务。",
  "has_person": true,
  "person_count": 1,
  "person_position_hint": "观测点前方",
  "key_objects": ["椅子", "桌子"],
  "advice": null
}

示例 3 — mode=find_person  waypoint=B
{
  "description": "当前近距离画面中可以确认前方有一名人员，位置在通道中央偏前方，身体轮廓比较完整，上半身和面向方向都较清楚。此人距离机器人较近，适合进行后续语音提示或交接动作。画面中没有看到第二个人，也没有发现人员被桌椅明显遮挡的情况。周围环境保持可通行状态，可以继续按任务流程靠近或等待对方响应。",
  "has_person": true,
  "person_count": 1,
  "person_position_hint": "通道中央",
  "key_objects": [],
  "advice": null
}

示例 3b — mode=find_person  waypoint=A   (画面中实际没人)
{
  "description": "当前近距离找人画面中暂未发现人员。画面中央、左右两侧以及靠近办公桌的可见区域内，都没有看到完整人形、清晰上半身或明显面部轮廓。当前结论只基于这一帧能看到的范围，若人员被桌椅、隔断或画面边缘遮挡，则无法在本次画面中确认。建议继续旋转扫描或前往下一个观测点补充搜索。",
  "has_person": false,
  "person_count": 0,
  "person_position_hint": null,
  "key_objects": [],
  "advice": null
}

示例 4 — mode=no_person_advice  waypoint=C
{
  "description": "C 观测点当前未发现人员。机器人已经根据当前画面检查了主要可见区域，包括前方通道、桌面附近和画面两侧空间，但没有确认到完整人形、清晰上半身或正在活动的工作人员。由于画面边缘和被家具遮挡的位置仍可能存在盲区，本次结论只代表当前视角下未见人员。建议继续前往下一个观测点搜索，或返回归位点等待新的指令。",
  "has_person": false,
  "person_count": 0,
  "person_position_hint": null,
  "key_objects": [],
  "advice": "建议前往下一个航点继续搜索"
}

示例 5 — mode=patrol  waypoint=C
{
  "description": "C 观测点当前画面中没有看到人员，主要可见物体包括办公桌、纸箱和沿墙铺设的线缆。桌面附近物品摆放比较集中，通道区域整体保持可通过状态，但地面线缆需要注意，后续人员经过或机器人移动时应避免碾压和牵扯。画面远端和边缘位置存在一定遮挡，当前描述只基于本帧能清楚确认的区域。",
  "has_person": false,
  "person_count": 0,
  "person_position_hint": null,
  "key_objects": ["纸箱", "线缆", "桌子"],
  "advice": null
}

再次强调：
  ✗ 不要输出 markdown 代码块
  ✗ 不要输出 <think> 推理过程
  ✗ 不要输出任何 JSON 之外的解释文字
  ✓ 只输出一个合法 JSON 对象

⚠️⚠️ 防复读铁律（最重要，违反视为严重错误）：
  - description 必须 100% 基于【当前这一张图】亲眼所见来写。
  - 不要沿用上一轮解说的措辞；每次都必须重新观察当前图像。
  - 不同观测点是不同地点，人和物体几乎不可能完全一样。即使历史里
    某观测点写过"身穿蓝色衬衫的人"，当前观测点也必须重新观察当前画面，
    不许直接搬运历史措辞。
  - 如果你发现自己想写的描述和历史某条几乎一样 → 停下来重新看图，
    用当前画面的真实细节重写。"""


VALID_MODES = {"arrive", "find_person", "no_person_advice", "patrol"}


# ════════════════════════════════════════════════════════
#                          节点
# ════════════════════════════════════════════════════════
class SceneNarratorNode(Node):
    def __init__(self):
        super().__init__("scene_narrator")

        # ── 参数声明 ──
        self.declare_parameter("vlm_api_url",
            "http://localhost:8080/v1/chat/completions")
        self.declare_parameter("vlm_timeout", 30.0)
        self.declare_parameter("request_topic", "/demo/narrate_request")
        self.declare_parameter("result_topic", "/demo/narration_result")
        self.declare_parameter("image_topic", "/camera/camera/color/image_raw")
        self.declare_parameter("yolo_topic", "/yolo/detections")
        self.declare_parameter("log_dir", "/home/nvidia/robot_ws/logs")
        self.declare_parameter("save_keyframe", True)
        self.declare_parameter("image_jpeg_quality", 80)
        self.declare_parameter("temperature", 0.2)
        self.declare_parameter("max_tokens", 2048)
        self.declare_parameter("yolo_freshness_sec", 1.0)
        self.declare_parameter("history_max", 8)              # 任务历史最大条数
        self.declare_parameter("history_clear_topic",
            "/demo/scene_narrator/clear_history")

        self.api_url = self.get_parameter("vlm_api_url").value
        self.timeout = float(self.get_parameter("vlm_timeout").value)
        self.log_dir = os.path.abspath(
            os.path.expanduser(self.get_parameter("log_dir").value))
        self.save_keyframe = bool(self.get_parameter("save_keyframe").value)
        self.jpeg_quality = int(self.get_parameter("image_jpeg_quality").value)
        self.temperature = float(self.get_parameter("temperature").value)
        self.max_tokens = int(self.get_parameter("max_tokens").value)
        self.yolo_freshness_sec = float(
            self.get_parameter("yolo_freshness_sec").value)
        history_max = int(self.get_parameter("history_max").value)
        clear_topic = self.get_parameter("history_clear_topic").value

        request_topic = self.get_parameter("request_topic").value
        result_topic = self.get_parameter("result_topic").value
        image_topic = self.get_parameter("image_topic").value
        yolo_topic = self.get_parameter("yolo_topic").value

        os.makedirs(self.log_dir, exist_ok=True)

        # ── 状态 ──
        self.bridge = CvBridge()
        self.latest_image = None
        self.image_lock = threading.Lock()
        self.latest_yolo = []      # [(class_zh, score, distance_m), ...]
        self.latest_yolo_time = 0.0
        self.yolo_lock = threading.Lock()
        self.is_inferring = False
        self.infer_lock = threading.Lock()

        # ── 任务历史（伪上下文）──
        # 节点启动即清（每次重启自然清空），跨 stage 不自动清，
        # 通过 /demo/scene_narrator/clear_history (Empty) 手动清。
        self.history = deque(maxlen=history_max)
        self.history_lock = threading.Lock()

        # ── ROS 接口 ──
        self.create_subscription(
            Image, image_topic, self._image_callback,
            qos_profile_sensor_data)
        self.create_subscription(
            Detection2DArray, yolo_topic, self._yolo_callback, 10)
        self.create_subscription(
            String, request_topic, self._request_callback, 10)
        # 清空历史
        self.create_subscription(
            Empty, clear_topic, self._clear_history_cb, 10)

        # 关键帧目标目录（latched 话题，由 mission_logger 在会话开始/结束时
        # 发布；不收时降级到 self.log_dir）
        latched_qos = QoSProfile(
            depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            history=QoSHistoryPolicy.KEEP_LAST)
        self.keyframe_dir_lock = threading.Lock()
        self.keyframe_dir = self.log_dir   # 默认值
        self.create_subscription(
            String, "/demo/log_dir",
            self._on_keyframe_dir, latched_qos)

        self.result_pub = self.create_publisher(String, result_topic, 10)

        # ── 启动日志 ──
        self.get_logger().info("════════════════════════════════════════")
        self.get_logger().info("🎙️ WL100 场景解说节点已启动")
        self.get_logger().info(f"  VLM API:    {self.api_url}")
        self.get_logger().info(f"  请求话题:   {request_topic}")
        self.get_logger().info(f"  结果话题:   {result_topic}")
        self.get_logger().info(f"  图像话题:   {image_topic}")
        self.get_logger().info(f"  YOLO 话题:  {yolo_topic}")
        self.get_logger().info(f"  日志目录:   {self.log_dir}")
        self.get_logger().info(
            f"  历史窗口:   {history_max} 条 "
            f"（重启自动清，或发 {clear_topic} 手动清）")
        self.get_logger().info(
            f"  支持模式:   {', '.join(sorted(VALID_MODES))}")
        self.get_logger().info("════════════════════════════════════════")
        self.get_logger().info(
            "💡 测试命令：ros2 topic pub --once /demo/narrate_request "
            "std_msgs/msg/String \"{data: '{\\\"mode\\\":\\\"arrive\\\","
            "\\\"waypoint\\\":\\\"A\\\"}'}\"")

    # ────────────────────────────────────────────────────
    #             图像缓存
    # ────────────────────────────────────────────────────
    def _image_callback(self, msg: Image):
        try:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            with self.image_lock:
                self.latest_image = img
        except Exception as e:
            self.get_logger().warn(f"图像转换失败: {e}",
                throttle_duration_sec=5.0)

    # ────────────────────────────────────────────────────
    #             YOLO 检测缓存（中文类名 + 置信度 + 深度）
    # ────────────────────────────────────────────────────
    def _yolo_callback(self, msg: Detection2DArray):
        seen = []
        for det in msg.detections:
            if not det.results:
                continue
            cls_en = det.results[0].hypothesis.class_id
            score = float(det.results[0].hypothesis.score)
            cls_zh = EN_TO_ZH.get(cls_en, cls_en)
            seen.append((cls_zh, score, _parse_detection_distance(det.id)))
        with self.yolo_lock:
            self.latest_yolo = seen
            self.latest_yolo_time = time.time()

    # ────────────────────────────────────────────────────
    #             清空历史
    # ────────────────────────────────────────────────────
    def _clear_history_cb(self, msg: Empty):
        with self.history_lock:
            n = len(self.history)
            self.history.clear()
        self.get_logger().info(f"🧹 已清空任务历史（清空前 {n} 条）")

    # ────────────────────────────────────────────────────
    #             关键帧目录切换
    # ────────────────────────────────────────────────────
    def _on_keyframe_dir(self, msg: String):
        path = (msg.data or "").strip()
        if not path:
            return
        try:
            os.makedirs(path, exist_ok=True)
        except Exception as e:
            self.get_logger().warn(f"创建关键帧目录失败 {path}: {e}")
            return
        with self.keyframe_dir_lock:
            old = self.keyframe_dir
            self.keyframe_dir = path
        if old != path:
            self.get_logger().info(f"📁 关键帧目录切到: {path}")

    # ────────────────────────────────────────────────────
    #             触发请求处理
    # ────────────────────────────────────────────────────
    def _request_callback(self, msg: String):
        raw = msg.data.strip()
        if not raw:
            return

        # 解析 mode + waypoint + task_title + user_request
        task_title = ""
        user_request = ""
        speech_id = ""
        try:
            req = json.loads(raw)
            mode = str(req.get("mode", "")).strip()
            waypoint = str(req.get("waypoint", "")).strip()
            task_title = str(req.get("task_title", "")).strip()
            user_request = str(req.get("user_request", "") or "").strip()
            speech_id = str(req.get("speech_id", "")).strip()
        except json.JSONDecodeError:
            mode = raw
            waypoint = ""

        if mode not in VALID_MODES:
            self.get_logger().warn(
                f"未知 mode: {mode!r}，忽略请求。支持: {sorted(VALID_MODES)}")
            return

        # 推理锁：同时只跑 1 个
        with self.infer_lock:
            if self.is_inferring:
                self.get_logger().warn(
                    f"上一次推理还在进行中，忽略新请求 (mode={mode}, wp={waypoint})")
                return
            self.is_inferring = True

        # 关键帧 + YOLO 列表 在主线程取（避免缓存被覆盖）
        with self.image_lock:
            frame = None if self.latest_image is None else self.latest_image.copy()

        with self.yolo_lock:
            yolo_age = time.time() - self.latest_yolo_time
            yolo_pairs = (list(self.latest_yolo)
                          if yolo_age < self.yolo_freshness_sec else [])

        if frame is None:
            self.get_logger().error("尚未收到任何图像帧，无法解说")
            with self.infer_lock:
                self.is_inferring = False
            return

        # 启动后台推理
        thread = threading.Thread(
            target=self._do_infer,
            args=(mode, waypoint, frame, yolo_pairs, task_title,
                  user_request, speech_id),
            daemon=True)
        thread.start()

    # ────────────────────────────────────────────────────
    #             推理主流程（后台线程）
    # ────────────────────────────────────────────────────
    def _do_infer(self, mode: str, waypoint: str, frame, yolo_pairs,
                  task_title: str = "", user_request: str = "",
                  speech_id: str = ""):
        t_start = time.time()
        now = datetime.now()
        weekday_zh = ["一", "二", "三", "四", "五", "六", "日"][now.weekday()]
        timestamp = (
            f"{now.year}年{now.month:02d}月{now.day:02d}日_周{weekday_zh}_"
            f"{now.hour:02d}点{now.minute:02d}分{now.second:02d}秒")
        keyframe_path = ""

        try:
            # 1. 关键帧落盘（推理之前先存，避免推理失败丢图）
            if self.save_keyframe:
                # 允许中文字符 + 字母数字下划线连字符（避免把"巡检A"改成"___A"）
                wp_safe = re.sub(r"[^\w\u4e00-\u9fff\-]", "_", waypoint) or "noname"
                keyframe_name = f"narrate_{timestamp}_{mode}_{wp_safe}.jpg"
                # 优先用 mission_logger 通知的会话目录；没收到时使用 log_dir 作为默认目录
                with self.keyframe_dir_lock:
                    target_dir = self.keyframe_dir
                try:
                    os.makedirs(target_dir, exist_ok=True)
                except Exception:
                    target_dir = self.log_dir
                keyframe_path = os.path.join(target_dir, keyframe_name)
                params = [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
                ok = cv2.imwrite(keyframe_path, frame, params)
                if not ok:
                    self.get_logger().warn(
                        f"关键帧写盘失败: {keyframe_path}")
                    keyframe_path = ""

            # 2. 图像 → base64
            params = [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
            ok, jpeg_buf = cv2.imencode(".jpg", frame, params)
            if not ok:
                raise RuntimeError("cv2.imencode 失败")
            data_uri = (
                "data:image/jpeg;base64,"
                + base64.b64encode(jpeg_buf).decode("utf-8"))

            # 3. YOLO 列表 → 结构化参考检测，仍由 VLM 以图片为主判断。
            yolo_items = []
            for cls_zh, score, distance_m in yolo_pairs[:8]:
                yolo_items.append({
                    "class": cls_zh,
                    "score": round(float(score), 2),
                    "distance_m": (
                        round(float(distance_m), 2)
                        if distance_m is not None else None
                    ),
                })
            yolo_str = json.dumps(yolo_items, ensure_ascii=False)

            # 4. 拼装 user 消息
            wp_show = waypoint if waypoint else "(未提供)"

            task_title_line = (
                f"任务标题：{task_title}\n" if task_title else "")
            user_request_line = (
                f"用户原始请求：{user_request}\n"
                if user_request else "")

            user_text = (
                f"{user_request_line}"
                f"{task_title_line}"
                f"当前位置：{wp_show}\n"
                f"模式：{mode}\n"
                f"视觉辅助检测：{yolo_str}\n\n"
                "请按 system 规则输出 JSON。")

            # DEBUG：打印 user_text，确认本轮只包含当前帧上下文
            self.get_logger().info(
                f"📤 user_text (前 600 字):\n{user_text[:600]}")

            # 5. POST llama-server
            payload = {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": [
                        {"type": "text", "text": user_text},
                        {"type": "image_url",
                         "image_url": {"url": data_uri}},
                    ]},
                ],
                "temperature": self.temperature,
                "top_k": 20,
                "top_p": 0.8,
                "max_tokens": self.max_tokens,
            }
            req_data = json.dumps(payload).encode("utf-8")
            req = Request(
                self.api_url,
                data=req_data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )

            with urlopen(req, timeout=self.timeout) as resp:
                resp_json = json.loads(resp.read().decode("utf-8"))

            raw_text = resp_json["choices"][0]["message"]["content"]

            # 6. 解析（含 fallback）
            parsed, fallback_flag = self._parse_response(raw_text, mode)

            # 7. 拼装结果
            result = {
                "trigger": {"mode": mode, "waypoint": waypoint},
                "speech_id": speech_id,
                "description": parsed["description"],
                "has_person": parsed["has_person"],
                "person_count": parsed["person_count"],
                "person_position_hint": parsed["person_position_hint"],
                "key_objects": parsed["key_objects"],
                "advice": parsed["advice"],
                "image_path": keyframe_path,
                "duration_sec": round(time.time() - t_start, 2),
                "fallback": fallback_flag,
            }

            # 发布
            out_msg = String()
            out_msg.data = json.dumps(result, ensure_ascii=False)
            self.result_pub.publish(out_msg)

            # 保留内部历史，供清空按钮/日志兼容；不再喂给 VLM。
            # 仅保留 trigger / description / has_person / person_position_hint
            # / key_objects / advice / fallback 这些字段
            with self.history_lock:
                self.history.append({
                    "trigger": result["trigger"],
                    "description": result["description"],
                    "has_person": result["has_person"],
                    "person_position_hint": result["person_position_hint"],
                    "key_objects": result["key_objects"],
                    "advice": result["advice"],
                    "fallback": result["fallback"],
                })

            self.get_logger().info(
                f"✅ 解说完成 ({result['duration_sec']:.1f}s, "
                f"mode={mode}, wp={waypoint}, "
                f"has_person={result['has_person']}"
                f"{', fallback' if fallback_flag else ''})")
            self.get_logger().info(f"   {result['description']}")

        except (HTTPError, URLError) as e:
            err_body = ""
            if isinstance(e, HTTPError):
                try:
                    err_body = e.read().decode("utf-8", errors="ignore")[:500]
                except Exception:
                    pass
            full_msg = f"{e}" + (f" body={err_body!r}" if err_body else "")
            self.get_logger().error(f"❌ HTTP 请求失败: {full_msg}")
            self._publish_error(mode, waypoint, keyframe_path,
                full_msg, t_start, speech_id)
        except Exception as e:
            self.get_logger().error(f"❌ 推理异常: {e}")
            self._publish_error(mode, waypoint, keyframe_path,
                f"Exception: {e}", t_start, speech_id)
        finally:
            with self.infer_lock:
                self.is_inferring = False

    # ────────────────────────────────────────────────────
    #             响应解析（含 fallback）
    # ────────────────────────────────────────────────────
    def _parse_response(self, raw_text: str, mode: str) -> tuple:
        """
        Returns: (parsed_dict, fallback_flag)
        """
        # 1. 剥离 <think>...</think>
        cleaned = re.sub(
            r"<think>.*?</think>", "", raw_text, flags=re.DOTALL).strip()
        # 2. 剥 markdown 代码块
        cleaned = re.sub(r"```json\s*", "", cleaned)
        cleaned = re.sub(r"```\s*", "", cleaned)
        cleaned = cleaned.strip()

        # 3. 提取最大 JSON 对象（贪心从第一个 { 到最后一个 }）
        first = cleaned.find("{")
        last = cleaned.rfind("}")
        if first < 0 or last <= first:
            return self._fallback_dict(raw_text, mode), True

        json_str = cleaned[first:last + 1]
        try:
            obj = json.loads(json_str)
        except json.JSONDecodeError:
            return self._fallback_dict(raw_text, mode), True

        # 4. 字段补全 + 类型校正
        return self._normalize_fields(obj, mode), False

    def _normalize_fields(self, obj: dict, mode: str) -> dict:
        """字段缺失/类型错误时填默认值，不抛异常"""
        result = {
            "description": "",
            "has_person": False,
            "person_count": 0,
            "person_position_hint": None,
            "key_objects": [],
            "advice": None,
        }

        # description
        d = obj.get("description")
        if isinstance(d, str):
            result["description"] = d.strip()

        # has_person
        hp = obj.get("has_person")
        if isinstance(hp, bool):
            result["has_person"] = hp
        elif isinstance(hp, str):
            result["has_person"] = hp.lower() in ("true", "yes", "有", "1")

        # person_count
        pc = obj.get("person_count", 0)
        try:
            result["person_count"] = max(0, int(pc))
        except (TypeError, ValueError):
            result["person_count"] = 1 if result["has_person"] else 0

        # person_position_hint
        pph = obj.get("person_position_hint")
        if isinstance(pph, str) and pph.strip():
            result["person_position_hint"] = pph.strip()

        # key_objects
        ko = obj.get("key_objects")
        if isinstance(ko, list):
            result["key_objects"] = [
                str(x).strip() for x in ko if str(x).strip()][:5]

        # advice
        ad = obj.get("advice")
        if isinstance(ad, str) and ad.strip():
            result["advice"] = ad.strip()

        # 后置约束修正：advice 仅在 no_person_advice 模式有效
        if mode != "no_person_advice":
            result["advice"] = None

        # has_person=false → person_count=0, position=null
        if not result["has_person"]:
            result["person_count"] = 0
            result["person_position_hint"] = None

        return result

    def _fallback_dict(self, raw_text: str, mode: str) -> dict:
        """完全解析失败时的格式保护处理"""
        # 从 raw_text 里硬抠 has_person
        has_person_guess = bool(re.search(
            r'"has_person"\s*:\s*true', raw_text, re.IGNORECASE))

        # 默认文案
        if mode == "arrive":
            desc = (
                "机器人已经到达当前航点，但本次 VLM 输出格式异常，无法可靠解析出完整的结构化结果。"
                "因此这里采用保守解说：当前只确认机器人已触发到达后的场景观察流程，"
                "画面中的具体人员、物体和风险点不做额外编造。请以现场画面和后续正常解说为准。")
        elif mode == "find_person":
            desc = (
                "机器人已经进入近距离找人解说流程，但本次 VLM 输出格式异常，无法可靠解析人员数量、"
                "方位和外观细节。为了避免误报，这里只说明流程已经触发，不额外编造人员特征、"
                "距离或动作状态。请结合当前画面、YOLO 检测和下一次正常解说继续判断。")
        elif mode == "no_person_advice":
            desc = (
                "本航点进入未发现人员后的建议流程，但本次 VLM 输出格式异常，无法可靠提取详细描述。"
                "因此这里采用保守结论：当前没有可解析的可靠人员信息，不额外编造人员位置或物体细节。"
                "建议继续前往下一个观测点搜索，或返回归位点等待新的人工指令。")
        elif mode == "patrol":
            desc = (
                "机器人正在执行该航点的巡检解说，但本次 VLM 输出格式异常，无法可靠解析物体列表、"
                "人员状态和现场建议。为避免错误描述，这里只确认巡检流程已触发，不补写画面中未能"
                "稳定确认的内容。请以当前实时画面、YOLO 检测和后续重新触发的解说结果为准。")
        else:
            desc = (
                "本次场景解说没有得到可解析的 VLM 结构化结果，因此无法给出可靠的人员、物体和建议字段。"
                "为避免误导，这里不编造画面内容，只说明解说流程异常结束。请检查 VLM 服务、相机画面和"
                "检测输入状态，必要时重新触发一次场景解说。")

        result = {
            "description": desc,
            "has_person": has_person_guess,
            "person_count": 1 if has_person_guess else 0,
            "person_position_hint": None,
            "key_objects": [],
            "advice": (
                "建议返回归位点等待下一条指令"
                if mode == "no_person_advice" else None),
        }
        return result

    # ────────────────────────────────────────────────────
    #             错误结果发布
    # ────────────────────────────────────────────────────
    def _publish_error(self, mode, waypoint, keyframe_path, err_msg, t_start,
                       speech_id: str = ""):
        result = {
            "trigger": {"mode": mode, "waypoint": waypoint},
            "speech_id": speech_id,
            "description": f"解说失败: {err_msg}",
            "has_person": False,
            "person_count": 0,
            "person_position_hint": None,
            "key_objects": [],
            "advice": None,
            "image_path": keyframe_path,
            "duration_sec": round(time.time() - t_start, 2),
            "fallback": True,
            "error": err_msg,
        }
        out_msg = String()
        out_msg.data = json.dumps(result, ensure_ascii=False)
        self.result_pub.publish(out_msg)


def main(args=None):
    rclpy.init(args=args)
    node = SceneNarratorNode()
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
