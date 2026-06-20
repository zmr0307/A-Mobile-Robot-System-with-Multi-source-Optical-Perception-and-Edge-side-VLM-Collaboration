#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WL100 语义目标匹配节点（人 + 物品，保留旧 person 接口兼容）

职责：
  - 缓存当前 RGB 图像和 YOLO 检测框
  - 接收 /demo/target_match_request（兼容 /demo/person_match_request）
  - 给 YOLO 候选框编号 P1/B1/C1/...
  - 把标注图 + 用户目标描述发给 VLM
  - 发布 /demo/target_match_result（兼容 /demo/person_match_result）

这个节点只做语义判断，不发布 cmd_vel，不发导航目标。
"""

import base64
import json
import math
import os
import re
import threading
import time
from datetime import datetime
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import String
from vision_msgs.msg import Detection2DArray

from wl100_demo.semantic_match_prompt import (
    MATCH_PROMPT,
    OBSERVATION_PROMPT,
    REQUIREMENT_PROMPT,
    SYSTEM_PROMPT,
    build_observation_user_text,
    build_requirement_user_text,
    build_semantic_match_user_text,
)


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


def _clean_json_text(raw_text: str) -> str:
    cleaned = re.sub(r"<think>.*?</think>", "", raw_text or "",
                     flags=re.DOTALL).strip()
    cleaned = re.sub(r"```json\s*", "", cleaned)
    cleaned = re.sub(r"```\s*", "", cleaned)
    return cleaned.strip()


def _normalise_target_type(value):
    raw = str(value or "").strip().lower()
    if not raw or raw in {"null", "none", "无", "没有"}:
        return None
    aliases = {
        "person": "person",
        "people": "person",
        "human": "person",
        "人": "person",
        "人员": "person",
        "工作人员": "person",
        "员工": "person",
        "cardboard box": "cardboard box",
        "box": "cardboard box",
        "carton": "cardboard box",
        "纸箱": "cardboard box",
        "箱子": "cardboard box",
        "盒子": "cardboard box",
        "chair": "chair",
        "椅子": "chair",
        "table": "table",
        "desk": "table",
        "桌子": "table",
        "桌": "table",
        "door": "door",
        "门": "door",
        "bag": "bag",
        "包": "bag",
        "backpack": "bag",
        "背包": "bag",
        "phone": "phone",
        "mobile phone": "phone",
        "手机": "phone",
    }
    if raw in aliases:
        return aliases[raw]
    return _infer_target_type_from_text(raw) or raw


def _infer_target_type_from_text(text):
    raw = str(text or "")
    patterns = [
        ("person", r"工作人员|员工|人员|目标人员|目标人|(?<!机器)(?<!无)人"),
        ("cardboard box", r"纸箱|箱子|盒子|box|carton"),
        ("chair", r"椅子|chair"),
        ("table", r"桌子|桌|table|desk"),
        ("bag", r"背包|包|bag|backpack"),
        ("phone", r"手机|phone"),
        ("door", r"门|door"),
    ]
    matches = []
    for target_type, pattern in patterns:
        for match in re.finditer(pattern, raw, re.IGNORECASE):
            matches.append((match.start(), match.end(), target_type))
    if not matches:
        return None
    matches.sort(key=lambda item: (item[1], item[0]))
    return matches[-1][2]


def _target_type_label(target_type):
    labels = {
        "person": "目标人",
        "cardboard box": "箱子",
        "chair": "椅子",
        "table": "桌子",
        "door": "门",
        "bag": "包",
        "phone": "手机",
    }
    return labels.get(target_type or "", "")


def _class_matches_target(class_name, target_type):
    if not target_type:
        return True
    return _normalise_target_type(class_name) == _normalise_target_type(
        target_type)


class SemanticTargetMatcherNode(Node):
    def __init__(self, node_name: str = "semantic_target_matcher"):
        super().__init__(node_name)

        self.declare_parameter("request_topic", "/demo/target_match_request")
        self.declare_parameter("result_topic", "/demo/target_match_result")
        self.declare_parameter("image_topic", "/camera/camera/color/image_raw")
        self.declare_parameter("yolo_topic", "/yolo/detections")
        self.declare_parameter(
            "vlm_api_url", "http://localhost:8080/v1/chat/completions")
        self.declare_parameter("vlm_timeout", 45.0)
        self.declare_parameter("temperature", 0.1)
        self.declare_parameter("max_tokens", 1024)
        self.declare_parameter("image_jpeg_quality", 85)
        self.declare_parameter("yolo_freshness_sec", 1.5)
        self.declare_parameter("candidate_min_score", 0.3)
        self.declare_parameter(
            "candidate_class_names",
            ["person", "人", "cardboard box", "纸箱", "箱子",
             "chair", "椅子", "table", "桌子", "door", "门",
             "bag", "包", "backpack", "背包", "phone", "手机"])
        self.declare_parameter("person_min_score", 0.3)
        self.declare_parameter("person_class_names", ["person", "人"])
        self.declare_parameter("max_candidates", 5)
        self.declare_parameter("match_min_confidence", 0.45)
        self.declare_parameter("save_debug_image", True)
        self.declare_parameter(
            "log_dir", "/home/nvidia/robot_ws/logs/target_match")

        request_topic = self.get_parameter("request_topic").value
        result_topic = self.get_parameter("result_topic").value
        image_topic = self.get_parameter("image_topic").value
        yolo_topic = self.get_parameter("yolo_topic").value

        self.api_url = self.get_parameter("vlm_api_url").value
        self.timeout = float(self.get_parameter("vlm_timeout").value)
        self.temperature = float(self.get_parameter("temperature").value)
        self.max_tokens = int(self.get_parameter("max_tokens").value)
        self.jpeg_quality = int(self.get_parameter("image_jpeg_quality").value)
        self.yolo_freshness_sec = float(
            self.get_parameter("yolo_freshness_sec").value)
        self.candidate_min_score = float(
            self.get_parameter("candidate_min_score").value)
        candidate_classes = list(
            self.get_parameter("candidate_class_names").value)
        if not candidate_classes:
            candidate_classes = list(
                self.get_parameter("person_class_names").value)
        self.candidate_classes = {str(v) for v in candidate_classes}
        self.max_candidates = int(self.get_parameter("max_candidates").value)
        self.match_min_confidence = float(
            self.get_parameter("match_min_confidence").value)
        self.save_debug_image = bool(
            self.get_parameter("save_debug_image").value)
        self.log_dir = os.path.abspath(os.path.expanduser(
            self.get_parameter("log_dir").value))
        os.makedirs(self.log_dir, exist_ok=True)

        self.bridge = CvBridge()
        self.image_lock = threading.Lock()
        self.latest_image = None
        self.latest_image_time = 0.0

        self.yolo_lock = threading.Lock()
        self.latest_candidates = []
        self.latest_yolo_time = 0.0

        self.infer_lock = threading.Lock()
        self.is_inferring = False

        self.create_subscription(
            Image, image_topic, self._image_callback, qos_profile_sensor_data)
        self.create_subscription(
            Detection2DArray, yolo_topic, self._yolo_callback, 10)
        self.create_subscription(
            String, request_topic, self._request_callback, 10)
        self.result_pub = self.create_publisher(String, result_topic, 10)

        self.get_logger().info("════════════════════════════════════════")
        self.get_logger().info("🎯 WL100 语义目标匹配节点已启动")
        self.get_logger().info(f"  VLM API:    {self.api_url}")
        self.get_logger().info(f"  请求话题:   {request_topic}")
        self.get_logger().info(f"  结果话题:   {result_topic}")
        self.get_logger().info(f"  图像话题:   {image_topic}")
        self.get_logger().info(f"  YOLO 话题:  {yolo_topic}")
        self.get_logger().info(
            f"  候选类别:   {sorted(self.candidate_classes) or ['全部']}")
        self.get_logger().info(
            f"  候选阈值:   {self.candidate_min_score:.2f}, "
            f"候选上限: {self.max_candidates}, "
            f"匹配阈值: {self.match_min_confidence:.2f}")
        self.get_logger().info(f"  调试图目录: {self.log_dir}")
        self.get_logger().info("════════════════════════════════════════")

    def _image_callback(self, msg: Image):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as exc:
            self.get_logger().warn(
                f"图像转换失败: {exc}", throttle_duration_sec=5.0)
            return
        with self.image_lock:
            self.latest_image = frame
            self.latest_image_time = time.time()

    def _yolo_callback(self, msg: Detection2DArray):
        candidates = []
        for det in msg.detections:
            if not det.results:
                continue
            cls = str(det.results[0].hypothesis.class_id)
            score = float(det.results[0].hypothesis.score)
            if (self.candidate_classes and cls not in self.candidate_classes):
                continue
            if score < self.candidate_min_score:
                continue
            cx = float(det.bbox.center.position.x)
            cy = float(det.bbox.center.position.y)
            w = float(det.bbox.size_x)
            h = float(det.bbox.size_y)
            if w <= 1.0 or h <= 1.0:
                continue
            candidates.append({
                "class": cls,
                "score": score,
                "cx": cx,
                "cy": cy,
                "w": w,
                "h": h,
                "area": w * h,
                "distance_m": _parse_detection_distance(det.id),
            })
            # 同类候选按画面从左到右编号，避免 B1/B2 只按面积排序导致
            # “左边/右边”描述和编号顺序不一致。
            candidates.sort(key=lambda item: (
                self._candidate_prefix(str(item.get("class", ""))),
                float(item.get("cx", 0.0)),
                float(item.get("cy", 0.0)),
                -float(item.get("area", 0.0)),
                -float(item.get("score", 0.0)),
            ))
            candidates = candidates[:max(1, self.max_candidates)]
        with self.yolo_lock:
            self.latest_candidates = candidates
            self.latest_yolo_time = time.time()

    def _request_callback(self, msg: String):
        raw = (msg.data or "").strip()
        if not raw:
            return
        try:
            req = json.loads(raw)
        except json.JSONDecodeError:
            self._publish_result({
                "request_id": "",
                "match": False,
                "person_id": None,
                "confidence": 0.0,
                "reason": "请求不是合法 JSON",
                "requirements": [],
                "failed_requirements": [],
                "uncertain_requirements": [],
                "exclude_ids": [],
                "candidates": [],
                "fallback": True,
            })
            return

        target_description = str(
            req.get("target_description", "") or "").strip()
        target_type = _normalise_target_type(
            req.get("target_type")
            or req.get("target_class")
            or req.get("type")
            or target_description)
        user_request = str(req.get("user_request", "") or "").strip()
        request_id = str(req.get("request_id", "") or "").strip()
        excluded_candidates = req.get("excluded_candidates") or []
        if not isinstance(excluded_candidates, list):
            excluded_candidates = []
        if not request_id:
            request_id = f"pm-{int(time.time() * 1000)}"

        if not target_description:
            target_description = _target_type_label(target_type)
        if not target_description:
            self._publish_result({
                "request_id": request_id,
                "match": False,
                "person_id": None,
                "target_type": target_type,
                "confidence": 0.0,
                "reason": "缺少 target_description",
                "requirements": [],
                "failed_requirements": [],
                "uncertain_requirements": [],
                "exclude_ids": [],
                "candidates": [],
                "fallback": True,
            })
            return

        with self.infer_lock:
            if self.is_inferring:
                self._publish_result({
                    "request_id": request_id,
                    "match": False,
                    "person_id": None,
                    "target_type": target_type,
                    "confidence": 0.0,
                    "reason": "上一轮目标语义匹配仍在推理中",
                    "requirements": [],
                    "failed_requirements": [],
                    "uncertain_requirements": [],
                    "exclude_ids": [],
                    "candidates": [],
                    "busy": True,
                    "fallback": True,
                })
                return
            self.is_inferring = True

        with self.image_lock:
            frame = None if self.latest_image is None else self.latest_image.copy()

        with self.yolo_lock:
            yolo_age = time.time() - self.latest_yolo_time
            candidates = (
                list(self.latest_candidates)
                if yolo_age <= self.yolo_freshness_sec else []
            )

        thread = threading.Thread(
            target=self._do_match,
            args=(request_id, target_type, target_description, user_request,
                  frame, candidates, yolo_age, excluded_candidates),
            daemon=True)
        thread.start()

    def _do_match(self, request_id: str, target_type: str | None,
                  target_description: str, user_request: str,
                  frame, candidates, yolo_age: float,
                  excluded_candidates: list):
        t0 = time.time()
        debug_image_path = ""
        candidate_payload = []
        try:
            if frame is None:
                self._publish_result({
                    "request_id": request_id,
                    "match": False,
                    "person_id": None,
                    "target_type": target_type,
                    "confidence": 0.0,
                    "reason": "尚未收到 RGB 图像",
                    "requirements": [],
                    "failed_requirements": [],
                    "uncertain_requirements": [],
                    "exclude_ids": [],
                    "candidates": [],
                    "duration_sec": round(time.time() - t0, 2),
                    "fallback": True,
                })
                return

            if not candidates:
                reason = "当前没有新鲜的 YOLO 候选目标"
                if yolo_age > self.yolo_freshness_sec:
                    reason += f"，最近检测已过期 {yolo_age:.1f}s"
                self._publish_result({
                    "request_id": request_id,
                    "match": False,
                    "person_id": None,
                    "target_type": target_type,
                    "confidence": 0.0,
                    "reason": reason,
                    "requirements": [],
                    "failed_requirements": [],
                    "uncertain_requirements": [],
                    "exclude_ids": [],
                    "candidates": [],
                    "duration_sec": round(time.time() - t0, 2),
                    "fallback": False,
                })
                return

            annotated = frame.copy()
            h_img, w_img = annotated.shape[:2]
            id_counts = {}
            excluded_count = 0
            for cand in candidates:
                target_class = str(cand.get("class", "") or "")
                x1, y1, x2, y2 = self._bbox_to_xyxy(cand, w_img, h_img)
                if self._candidate_is_excluded(
                        target_class, [x1, y1, x2, y2],
                        excluded_candidates):
                    excluded_count += 1
                    continue
                prefix = self._candidate_prefix(target_class)
                id_counts[prefix] = id_counts.get(prefix, 0) + 1
                target_id = f"{prefix}{id_counts[prefix]}"
                person_id = (
                    target_id if self._is_person_class(target_class) else None
                )
                cand["target_id"] = target_id
                cand["target_class"] = target_class
                cand["person_id"] = person_id
                cand["bbox_xyxy"] = [x1, y1, x2, y2]
                candidate_payload.append({
                    "target_id": target_id,
                    "target_class": target_class,
                    "person_id": person_id,
                    "bbox": [x1, y1, x2, y2],
                    "score": round(float(cand["score"]), 2),
                    "distance_m": (
                        round(float(cand["distance_m"]), 2)
                        if cand["distance_m"] is not None else None
                    ),
                })
                self._draw_candidate(annotated, target_id, x1, y1, x2, y2)

            if not candidate_payload:
                reason = "当前没有可送检的 YOLO 候选目标"
                if excluded_count:
                    reason = (
                        f"当前 {excluded_count} 个候选均已在本轮被否决")
                self._publish_result({
                    "request_id": request_id,
                    "match": False,
                    "person_id": None,
                    "target_type": target_type,
                    "confidence": 0.0,
                    "reason": reason,
                    "requirements": [],
                    "failed_requirements": [],
                    "uncertain_requirements": [],
                    "exclude_ids": [],
                    "candidates": [],
                    "duration_sec": round(time.time() - t0, 2),
                    "fallback": False,
                })
                return

            if self.save_debug_image:
                debug_image_path = self._save_debug_image(
                    annotated, request_id)

            data_uri = self._encode_image(annotated)

            t_observation = time.time()
            raw_observation = self._call_vlm_with_prompt(
                OBSERVATION_PROMPT,
                build_observation_user_text(candidate_payload),
                data_uri=data_uri)
            observation_sec = time.time() - t_observation
            visual_facts = self._parse_aux_json(
                raw_observation, "visual_observation")
            self.get_logger().info(
                f"语义匹配阶段1/3 视觉观察完成: "
                f"{observation_sec:.1f}s, request_id={request_id}")
            self.get_logger().info(
                "语义匹配阶段1输出 raw_observation:\n"
                f"{self._clip_log_text(raw_observation)}")

            t_requirements = time.time()
            raw_requirements = self._call_vlm_with_prompt(
                REQUIREMENT_PROMPT,
                build_requirement_user_text(
                    target_type, target_description, user_request),
                data_uri=None)
            requirements_sec = time.time() - t_requirements
            visual_requirements = self._parse_aux_json(
                raw_requirements, "visual_requirements")
            self.get_logger().info(
                f"语义匹配阶段2/3 需求拆解完成: "
                f"{requirements_sec:.1f}s, request_id={request_id}")
            self.get_logger().info(
                "语义匹配阶段2输出 raw_requirements:\n"
                f"{self._clip_log_text(raw_requirements)}")

            user_text = self._build_user_text(
                target_type, target_description, user_request,
                candidate_payload, visual_facts, visual_requirements)
            t_match = time.time()
            raw_text = self._call_vlm_with_prompt(
                MATCH_PROMPT, user_text, data_uri=data_uri)
            match_sec = time.time() - t_match
            self.get_logger().info(
                f"语义匹配阶段3/3 最终判断完成: "
                f"{match_sec:.1f}s, request_id={request_id}")
            self.get_logger().info(
                "语义匹配阶段3输出 raw_vlm:\n"
                f"{self._clip_log_text(raw_text)}")
            parsed = self._parse_response(raw_text, candidate_payload)
            spatial_reason = self._try_simple_spatial_match(
                parsed, target_description, target_type,
                candidate_payload, w_img)
            fallback = False
            reason = parsed["reason"]
            if spatial_reason:
                reason = spatial_reason

            if parsed["failed_requirements"]:
                parsed["match"] = False
                parsed["target_id"] = None
                parsed["target_class"] = None
                parsed["person_id"] = None
                parsed["answer"] = "不是"
                failed = "、".join(parsed["failed_requirements"])
                reason = f"存在不满足的明确条件：{failed}；{reason}"
            elif parsed["uncertain_requirements"]:
                parsed["match"] = False
                parsed["target_id"] = None
                parsed["target_class"] = None
                parsed["person_id"] = None
                parsed["answer"] = "不是"
                uncertain = "、".join(parsed["uncertain_requirements"])
                reason = f"存在无法确认的明确条件：{uncertain}；{reason}"

            if (parsed["match"]
                    and not _class_matches_target(
                        parsed["target_class"], target_type)):
                parsed["match"] = False
                parsed["target_id"] = None
                parsed["target_class"] = None
                parsed["person_id"] = None
                parsed["answer"] = "不是"
                reason = (
                    "VLM 选择的候选类别与目标类别不一致；"
                    f"{reason}")

            if parsed["match"] and parsed["confidence"] < self.match_min_confidence:
                parsed["match"] = False
                parsed["target_id"] = None
                parsed["target_class"] = None
                parsed["person_id"] = None
                parsed["answer"] = "不是"
                reason = (
                    f"VLM 置信度 {parsed['confidence']:.2f} "
                    f"低于阈值 {self.match_min_confidence:.2f}；"
                    f"{reason}")

            result = {
                "request_id": request_id,
                "match": parsed["match"],
                "target_id": parsed["target_id"],
                "target_class": parsed["target_class"],
                "person_id": parsed["person_id"],
                "target_type": target_type,
                "answer": parsed["answer"],
                "confidence": parsed["confidence"],
                "reason": reason,
                "requirements": parsed["requirements"],
                "failed_requirements": parsed["failed_requirements"],
                "uncertain_requirements": parsed["uncertain_requirements"],
                "exclude_ids": parsed["exclude_ids"],
                "referenced_objects": parsed["referenced_objects"],
                "candidate_assessments": parsed["candidate_assessments"],
                "target_description": target_description,
                "user_request": user_request,
                "candidates": candidate_payload,
                "image_path": debug_image_path,
                "duration_sec": round(time.time() - t0, 2),
                "stage_durations_sec": {
                    "observation": round(observation_sec, 2),
                    "requirements": round(requirements_sec, 2),
                    "match": round(match_sec, 2),
                },
                "fallback": fallback,
                "visual_facts": visual_facts,
                "visual_requirements": visual_requirements,
                "raw_observation": raw_observation[:500],
                "raw_requirements": raw_requirements[:500],
                "raw_vlm": raw_text[:500],
            }
            self._publish_result(result)
            self.get_logger().info(
                f"✅ 语义目标匹配完成: match={result['match']}, "
                f"target_id={result['target_id']}, "
                f"target_class={result['target_class']}, "
                f"person_id={result['person_id']}, "
                f"conf={result['confidence']:.2f}, "
                f"reason={result['reason']}")
        except (HTTPError, URLError, TimeoutError) as exc:
            self._publish_error(
                request_id, target_type, target_description, user_request,
                candidate_payload, debug_image_path, t0,
                f"VLM 请求失败: {exc}")
        except Exception as exc:
            self._publish_error(
                request_id, target_type, target_description, user_request,
                candidate_payload, debug_image_path, t0,
                f"推理异常: {exc}")
        finally:
            with self.infer_lock:
                self.is_inferring = False

    @staticmethod
    def _bbox_to_xyxy(cand, w_img: int, h_img: int):
        x1 = int(round(cand["cx"] - cand["w"] / 2.0))
        y1 = int(round(cand["cy"] - cand["h"] / 2.0))
        x2 = int(round(cand["cx"] + cand["w"] / 2.0))
        y2 = int(round(cand["cy"] + cand["h"] / 2.0))
        x1 = max(0, min(w_img - 1, x1))
        y1 = max(0, min(h_img - 1, y1))
        x2 = max(0, min(w_img - 1, x2))
        y2 = max(0, min(h_img - 1, y2))
        return x1, y1, x2, y2

    @staticmethod
    def _bbox_iou(a, b) -> float:
        try:
            ax1, ay1, ax2, ay2 = [float(v) for v in a]
            bx1, by1, bx2, by2 = [float(v) for v in b]
        except (TypeError, ValueError):
            return 0.0
        inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
        inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
        inter = inter_w * inter_h
        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        denom = area_a + area_b - inter
        return inter / denom if denom > 0 else 0.0

    @staticmethod
    def _bbox_center_distance_ratio(a, b) -> float:
        try:
            ax1, ay1, ax2, ay2 = [float(v) for v in a]
            bx1, by1, bx2, by2 = [float(v) for v in b]
        except (TypeError, ValueError):
            return 999.0
        acx, acy = (ax1 + ax2) / 2.0, (ay1 + ay2) / 2.0
        bcx, bcy = (bx1 + bx2) / 2.0, (by1 + by2) / 2.0
        scale = max(ax2 - ax1, ay2 - ay1, bx2 - bx1, by2 - by1, 1.0)
        return math.hypot(acx - bcx, acy - bcy) / scale

    def _candidate_is_excluded(self, target_class: str, bbox: list,
                               excluded_candidates: list) -> bool:
        if not excluded_candidates:
            return False
        if _normalise_target_type(target_class) == "person":
            return False
        for item in excluded_candidates:
            if not isinstance(item, dict):
                continue
            excluded_class = item.get("target_class") or item.get("class")
            excluded_bbox = item.get("bbox")
            if (not isinstance(excluded_bbox, list)
                    or len(excluded_bbox) != 4
                    or not _normalise_target_type(excluded_class)
                    or _normalise_target_type(excluded_class)
                    != _normalise_target_type(target_class)):
                continue
            if self._bbox_iou(bbox, excluded_bbox) >= 0.35:
                return True
            if self._bbox_center_distance_ratio(bbox, excluded_bbox) <= 0.18:
                return True
        return False

    @staticmethod
    def _candidate_prefix(class_name: str) -> str:
        cls = (class_name or "").strip().lower()
        mapping = {
            "person": "P",
            "人": "P",
            "cardboard box": "B",
            "纸箱": "B",
            "箱子": "B",
            "chair": "C",
            "椅子": "C",
            "table": "T",
            "桌子": "T",
            "door": "D",
            "门": "D",
            "bag": "G",
            "包": "G",
            "backpack": "G",
            "背包": "G",
            "phone": "M",
            "mobile phone": "M",
            "手机": "M",
            "blackboard": "W",
            "白板": "W",
        }
        return mapping.get(cls, "O")

    @staticmethod
    def _is_person_class(class_name: str) -> bool:
        return (class_name or "").strip().lower() in {"person", "人"}

    @staticmethod
    def _draw_candidate(img, target_id: str, x1: int, y1: int,
                        x2: int, y2: int):
        color = (0, 0, 255)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        label = target_id
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.9
        thickness = 2
        (tw, th), baseline = cv2.getTextSize(label, font, scale, thickness)
        y_text = max(th + baseline + 4, y1)
        cv2.rectangle(
            img,
            (x1, y_text - th - baseline - 6),
            (x1 + tw + 10, y_text + baseline),
            color,
            -1,
        )
        cv2.putText(
            img, label, (x1 + 5, y_text - 5), font, scale,
            (255, 255, 255), thickness, cv2.LINE_AA)

    def _save_debug_image(self, annotated, request_id: str) -> str:
        safe_id = re.sub(r"[^a-zA-Z0-9_\-]", "_", request_id)[:60] or "request"
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        path = os.path.join(self.log_dir, f"person_match_{ts}_{safe_id}.jpg")
        ok = cv2.imwrite(
            path, annotated, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
        return path if ok else ""

    def _encode_image(self, frame) -> str:
        ok, jpeg_buf = cv2.imencode(
            ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
        if not ok:
            raise RuntimeError("cv2.imencode 失败")
        return (
            "data:image/jpeg;base64,"
            + base64.b64encode(jpeg_buf).decode("utf-8")
        )

    @staticmethod
    def _build_user_text(target_type: str | None,
                         target_description: str,
                         user_request: str,
                         candidates: list[dict],
                         visual_facts: dict | None = None,
                         visual_requirements: dict | None = None) -> str:
        return build_semantic_match_user_text(
            target_type, target_description, user_request, candidates,
            visual_facts, visual_requirements)

    def _call_vlm(self, data_uri: str, user_text: str) -> str:
        return self._call_vlm_with_prompt(
            SYSTEM_PROMPT, user_text, data_uri=data_uri)

    def _call_vlm_with_prompt(self, system_prompt: str, user_text: str,
                              data_uri: str | None = None) -> str:
        user_content = user_text
        if data_uri:
            user_content = [
                {"type": "text", "text": user_text},
                {"type": "image_url", "image_url": {"url": data_uri}},
            ]
        payload = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            "temperature": self.temperature,
            "top_k": 20,
            "top_p": 0.8,
            "max_tokens": self.max_tokens,
        }
        req = Request(
            self.api_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(req, timeout=self.timeout) as resp:
            resp_json = json.loads(resp.read().decode("utf-8"))
        return resp_json["choices"][0]["message"]["content"]

    @staticmethod
    def _clip_log_text(text: str, limit: int = 5000) -> str:
        value = str(text or "").strip()
        if len(value) <= limit:
            return value
        return value[:limit] + f"\n...（已截断，原始长度 {len(value)} 字符）"

    @staticmethod
    def _parse_aux_json(raw_text: str, label: str) -> dict:
        cleaned = _clean_json_text(raw_text)
        first = cleaned.find("{")
        last = cleaned.rfind("}")
        if first < 0 or last <= first:
            return {
                "parse_error": f"{label} 未输出 JSON",
                "raw_text": cleaned[:1500],
            }
        try:
            obj = json.loads(cleaned[first:last + 1])
        except json.JSONDecodeError as exc:
            return {
                "parse_error": f"{label} JSON 解析失败: {exc}",
                "raw_text": cleaned[:1500],
            }
        if not isinstance(obj, dict):
            return {
                "parse_error": f"{label} JSON 不是对象",
                "raw_text": cleaned[:1500],
            }
        return obj

    def _parse_response(self, raw_text: str, candidates: list[dict]) -> dict:
        candidates_by_id = {}
        for item in candidates:
            target_id = item.get("target_id") or item.get("person_id")
            if target_id:
                candidates_by_id[str(target_id).strip().upper()] = item
        cleaned = _clean_json_text(raw_text)
        first = cleaned.find("{")
        last = cleaned.rfind("}")
        if first < 0 or last <= first:
            return {
                "match": False,
                "target_id": None,
                "target_class": None,
                "person_id": None,
                "answer": "不是",
                "confidence": 0.0,
                "requirements": [],
                "failed_requirements": [],
                "uncertain_requirements": [],
                "exclude_ids": [],
                "referenced_objects": [],
                "candidate_assessments": [],
                "reason": "VLM 未输出 JSON",
            }
        try:
            obj = json.loads(cleaned[first:last + 1])
        except json.JSONDecodeError:
            return {
                "match": False,
                "target_id": None,
                "target_class": None,
                "person_id": None,
                "answer": "不是",
                "confidence": 0.0,
                "requirements": [],
                "failed_requirements": [],
                "uncertain_requirements": [],
                "exclude_ids": [],
                "referenced_objects": [],
                "candidate_assessments": [],
                "reason": "VLM JSON 解析失败",
            }

        match = obj.get("match")
        if isinstance(match, str):
            match = match.lower() in ("true", "yes", "1", "是", "匹配")
        else:
            match = bool(match)

        target_id = obj.get("target_id") or obj.get("person_id")
        target_id = str(target_id).strip().upper() if target_id else None
        if target_id in ("NULL", "NONE", "无", ""):
            target_id = None

        selected = candidates_by_id.get(target_id) if target_id else None
        target_class = (
            str(obj.get("target_class", "") or "").strip()
            or (selected or {}).get("target_class")
            or (selected or {}).get("class")
            or None
        )
        person_id = (selected or {}).get("person_id") if selected else None
        answer = str(obj.get("answer", "") or "").strip()
        if not answer:
            answer = "是" if match else "不是"
        if answer.startswith("不是") or answer.lower() in {"no", "false"}:
            match = False

        try:
            confidence = float(obj.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        confidence = max(0.0, min(1.0, confidence))

        reason = str(obj.get("reason", "") or "").strip()
        if not reason:
            reason = "VLM 未给出原因"

        requirements = self._normalise_requirements(obj.get("requirements"))
        failed_requirements = self._normalise_string_list(
            obj.get("failed_requirements"))
        uncertain_requirements = self._normalise_string_list(
            obj.get("uncertain_requirements"))
        referenced_objects = self._normalise_referenced_objects(
            obj.get("referenced_objects"))
        candidate_assessments = self._normalise_candidate_assessments(
            obj.get("candidate_assessments"), candidates_by_id)
        exclude_ids = self._normalise_candidate_id_list(
            obj.get("exclude_ids"), candidates_by_id)
        for item in requirements:
            req = item.get("requirement", "")
            status = item.get("status", "")
            evidence = item.get("evidence", "")
            if status == "true" and self._contains_negative_evidence(evidence):
                item["status"] = "false"
                status = "false"
                if req and req not in failed_requirements:
                    failed_requirements.append(req)
            if status == "false" and req and req not in failed_requirements:
                failed_requirements.append(req)
            elif (status == "uncertain" and req
                  and req not in uncertain_requirements):
                uncertain_requirements.append(req)

        contradiction = self._find_obvious_contradiction(
            requirements, reason)
        if contradiction:
            match = False
            answer = "不是"
            if contradiction not in failed_requirements:
                failed_requirements.append(contradiction)
            reason = f"VLM 输出存在明显矛盾：{contradiction}；{reason}"

            semantic_contradiction = self._find_match_semantic_contradiction(
                match, target_id, reason, requirements, candidate_assessments)
            if semantic_contradiction:
                match = False
                answer = "不是"
                if semantic_contradiction not in failed_requirements:
                    failed_requirements.append(semantic_contradiction)
                reason = f"VLM 输出自相矛盾：{semantic_contradiction}；{reason}"

            visual_fact_conflict = self._find_visual_fact_conflict(
                match, target_id, target_description,
                visual_requirements, visual_facts)
            if visual_fact_conflict:
                match = False
                answer = "不是"
                if visual_fact_conflict not in failed_requirements:
                    failed_requirements.append(visual_fact_conflict)
                reason = f"第一阶段视觉事实与目标条件冲突：{visual_fact_conflict}；{reason}"

            reference_override = self._resolve_reference_relation(
                match, target_id, target_type, target_description,
                visual_requirements, visual_facts, candidate_payload, reason)
            if reference_override.get("conflict"):
                match = False
                answer = "不是"
                if reference_override["conflict"] not in failed_requirements:
                    failed_requirements.append(reference_override["conflict"])
                reason = (
                    f"参照物关系证据不足或冲突："
                    f"{reference_override['conflict']}；{reason}")
            elif reference_override.get("override_target_id"):
                override_id = reference_override["override_target_id"]
                override_candidate = next(
                    (item for item in candidate_payload
                     if str(item.get("target_id", "")).strip().upper()
                     == override_id),
                    None)
                if override_candidate:
                    match = True
                    answer = "是"
                    target_id = override_id
                    target_class = override_candidate.get("target_class")
                    person_id = override_candidate.get("person_id")
                    confidence = max(confidence, 0.72)
                    reason = (
                        f"{reference_override['override_reason']}；"
                        f"{reason}")

        if target_id not in candidates_by_id:
            if match:
                reason = f"VLM 返回了无效 target_id={target_id!r}；{reason}"
            match = False
            target_id = None
            target_class = None
            person_id = None

        if target_id and target_id in exclude_ids:
            exclude_ids = [item for item in exclude_ids if item != target_id]

        if not match:
            target_id = None
            target_class = None
            person_id = None
            answer = "不是"

        return {
            "match": match,
            "target_id": target_id,
            "target_class": target_class,
            "person_id": person_id,
            "answer": answer,
            "confidence": confidence,
            "requirements": requirements,
            "failed_requirements": failed_requirements,
            "uncertain_requirements": uncertain_requirements,
            "exclude_ids": exclude_ids,
            "referenced_objects": referenced_objects,
            "candidate_assessments": candidate_assessments,
            "reason": reason,
        }

    @classmethod
    def _find_match_semantic_contradiction(
            cls, match: bool, target_id: str | None, reason: str,
            requirements: list[dict],
            candidate_assessments: list[dict]) -> str:
        if not match:
            return ""
        if cls._contains_negative_evidence(reason):
            return "顶层原因包含不符合或无法确认的证据"
        ambiguity = cls._find_multi_candidate_ambiguity(
            reason, candidate_assessments)
        if ambiguity:
            return ambiguity
        for item in requirements:
            req = str(item.get("requirement", "") or "")
            evidence = str(item.get("evidence", "") or "")
            if item.get("status") == "true" and cls._contains_negative_evidence(
                    f"{req} {evidence}"):
                return f"条件 [{req}] 标为满足，但证据是否定或无法确认"
        if target_id:
            for item in candidate_assessments:
                cand_id = str(item.get("target_id", "") or "").strip().upper()
                if cand_id != target_id:
                    continue
                cand_reason = str(item.get("reason", "") or "")
                if not bool(item.get("meets")):
                    return f"候选 {target_id} 的逐项评估为不符合"
                if cls._contains_negative_evidence(cand_reason):
                    return f"候选 {target_id} 的原因包含不符合或无法确认的证据"
        return ""

    @classmethod
    def _find_visual_fact_conflict(
            cls, match: bool, target_id: str | None,
            target_description: str,
            visual_requirements: dict | None,
            visual_facts: dict | None) -> str:
        if not match or not target_id:
            return ""
        required_posture = cls._extract_required_posture(
            target_description, visual_requirements)
        if not required_posture:
            return ""
        observed_posture = cls._extract_candidate_posture(
            target_id, visual_facts)
        if not observed_posture:
            return ""
        if required_posture != observed_posture:
            return (
                f"用户要求{cls._posture_label(required_posture)}，"
                f"但第一阶段视觉事实显示候选为"
                f"{cls._posture_label(observed_posture)}"
            )
        return ""

    @classmethod
    def _resolve_reference_relation(
            cls, match: bool, target_id: str | None,
            target_type: str | None, target_description: str,
            visual_requirements: dict | None,
            visual_facts: dict | None,
            candidates: list[dict],
            current_reason: str) -> dict:
        refs = cls._extract_reference_constraints(
            target_description, visual_requirements)
        if not refs:
            return {}

        scores, found_reference = cls._score_candidates_by_reference(
            refs, target_type, visual_facts, candidates)
        if not found_reference:
            if match:
                names = "、".join(ref["name"] for ref in refs)
                return {
                    "conflict": f"未确认参照物 [{names}]",
                }
            return {}

        if not scores:
            if match:
                names = "、".join(ref["name"] for ref in refs)
                return {
                    "conflict": f"已看到参照物 [{names}]，但未建立到候选的明确邻近关系",
                }
            return {}

        best_score = max(scores.values())
        best_ids = sorted([cid for cid, score in scores.items()
                           if score == best_score])
        if len(best_ids) != 1:
            if match:
                return {
                    "conflict": "多个候选与参照物关系同样强，无法唯一确定",
                }
            return {}

        best_id = best_ids[0]
        if not match:
            return {}
        if not target_id:
            return {
                "override_target_id": best_id,
                "override_reason": f"根据参照物关系，唯一更符合的是 {best_id}",
            }
        if str(target_id).strip().upper() != best_id:
            return {
                "override_target_id": best_id,
                "override_reason": (
                    f"根据参照物关系，{best_id} 比 {target_id} 更靠近目标参照物"),
            }
        if cls._contains_negative_evidence(current_reason):
            return {
                "conflict": f"当前原因与参照物关系判断冲突，唯一更符合的是 {best_id}",
            }
        return {}

    @staticmethod
    def _find_multi_candidate_ambiguity(
            reason: str, candidate_assessments: list[dict]) -> str:
        true_ids = []
        for item in candidate_assessments:
            if bool(item.get("meets")):
                target_id = str(
                    item.get("target_id", "") or "").strip().upper()
                if target_id:
                    true_ids.append(target_id)
        if len(set(true_ids)) > 1:
            return "多个候选都被评估为符合，无法唯一确定目标"

        compact = re.sub(r"\s+", "", str(reason or "")).upper()
        ids = set(re.findall(r"\b[A-Z]\d+\b", compact))
        says_multi_true = any(
            word in compact
            for word in ("都满足", "都符合", "均满足", "均符合", "都相邻",
                         "都在旁边", "都靠近"))
        says_clear_best = any(
            word in compact
            for word in ("明显更近", "更符合", "最符合", "唯一符合",
                         "关系更强", "最接近"))
        if len(ids) > 1 and says_multi_true and not says_clear_best:
            return "VLM 原因称多个候选都符合，无法唯一确定目标"
        return ""

    @staticmethod
    def _contains_negative_evidence(text: str) -> bool:
        compact = re.sub(r"\s+", "", str(text or "")).lower()
        if not compact:
            return False
        negative_patterns = (
            "不符合", "不满足", "不是", "并非", "不能算",
            "无法确认", "不能确认", "未能确认", "无法判断",
            "证据不足", "关系不明确", "关系不足", "关系证据不足",
            "未看到", "没有看到", "看不到", "看不清", "未发现",
            "没有发现", "缺少", "不存在", "不明显", "不能证明",
            "无法说明", "无法证明", "不等于", "不足以",
            "notmatch", "notmatched", "notvisible", "uncertain",
            "unclear", "insufficient", "cannotconfirm", "can'tconfirm",
        )
        positive_exceptions = (
            "没有不符合", "没有不满足", "无不符合", "无不满足",
            "不存在不符合", "未发现不符合", "未发现不满足",
        )
        if any(item in compact for item in positive_exceptions):
            return False
        return any(item in compact for item in negative_patterns)

    @staticmethod
    def _try_simple_spatial_match(
            parsed: dict, target_description: str, target_type: str | None,
            candidates: list[dict], image_width: int) -> str:
        """纯左右/中间目标使用 bbox 几何一致性检查，避免 VLM 把方位词当名字。"""
        desc = re.sub(r"\s+", "", str(target_description or "").lower())
        if not desc:
            return ""
        wants_left = any(word in desc for word in ("左边", "最左", "左侧"))
        wants_right = any(word in desc for word in ("右边", "最右", "右侧"))
        wants_middle = any(word in desc for word in ("中间", "中间的", "中央"))
        if sum([wants_left, wants_right, wants_middle]) != 1:
            return ""

        # 只处理简单方位，不覆盖颜色、姿态、携带物、邻近关系等语义条件。
        complex_words = (
            "穿", "拿", "抱", "提", "背", "捧", "端", "坐", "站", "蹲",
            "旁边", "附近", "靠近", "挨着", "有", "红", "黑", "白",
            "蓝", "绿", "黄", "花盆", "椅子旁", "桌子旁")
        if any(word in desc for word in complex_words):
            return ""

        pool = []
        for cand in candidates:
            cls = cand.get("target_class") or cand.get("class")
            if target_type and not _class_matches_target(cls, target_type):
                continue
            bbox = cand.get("bbox") or []
            if len(bbox) != 4:
                continue
            try:
                x1, _y1, x2, _y2 = [float(v) for v in bbox]
            except (TypeError, ValueError):
                continue
            target_id = cand.get("target_id") or cand.get("person_id")
            if not target_id:
                continue
            pool.append((float((x1 + x2) / 2.0), cand))
        if not pool:
            return ""

        if wants_left:
            chosen = min(pool, key=lambda item: item[0])
            if len(pool) == 1 and chosen[0] > image_width * 0.55:
                return ""
        elif wants_right:
            chosen = max(pool, key=lambda item: item[0])
            if len(pool) == 1 and chosen[0] < image_width * 0.45:
                return ""
        else:
            mid = image_width / 2.0
            chosen = min(pool, key=lambda item: abs(item[0] - mid))

        cand = chosen[1]
        target_id = cand.get("target_id") or cand.get("person_id")
        target_class = cand.get("target_class") or cand.get("class")
        parsed["match"] = True
        parsed["target_id"] = target_id
        parsed["target_class"] = target_class
        parsed["person_id"] = cand.get("person_id")
        parsed["answer"] = "是"
        parsed["confidence"] = max(float(parsed.get("confidence", 0.0) or 0.0), 0.72)
        parsed["requirements"] = [{
            "requirement": target_description,
            "status": "true",
            "evidence": f"按候选框水平位置选择 {target_id}",
        }]
        parsed["failed_requirements"] = []
        parsed["uncertain_requirements"] = []
        parsed["exclude_ids"] = []
        return f"简单方位目标按候选框位置确认：{target_id} 符合 [{target_description}]"

    @staticmethod
    def _find_obvious_contradiction(
            requirements: list[dict], reason: str) -> str:
        text = " ".join(
            [str(reason or "")]
            + [
                f"{item.get('requirement', '')} {item.get('evidence', '')}"
                for item in requirements
                if isinstance(item, dict)
            ])
        compact = re.sub(r"\s+", "", text).lower()
        wants_standing = any(
            word in compact for word in ("站着", "站立", "直立", "standing"))
        says_sitting = any(
            word in compact
            for word in ("坐着", "坐在", "坐姿", "椅子上", "蹲着", "半蹲", "倚坐", "sitting"))
        if wants_standing and says_sitting:
            return "用户要求站着，但 VLM 证据描述为坐姿/坐在椅子上"
        wants_sitting = any(
            word in compact
            for word in ("坐着", "坐在", "坐姿", "椅子上", "sitting"))
        says_standing = any(
            word in compact for word in ("站着", "站立", "直立", "standing"))
        if wants_sitting and says_standing:
            return "用户要求坐着，但 VLM 证据描述为站立姿势"

        wants_holding = any(
            word in compact
            for word in ("拿箱子", "拿着箱子", "抱箱子", "抱着箱子",
                         "提箱子", "提着箱子", "拿纸箱", "抱纸箱",
                         "有箱子"))
        weak_relation = any(
            word in compact for word in ("旁边", "附近", "靠近", "同一画面"))
        strong_holding = any(
            word in compact
            for word in ("拿着", "抱着", "提着", "端着", "背着",
                         "捧着", "手中", "怀里", "稳定接触", "持有"))
        if wants_holding and weak_relation and not strong_holding:
            return "用户要求拿/抱箱子，但 VLM 证据只描述旁边/附近关系"
        return ""

    @staticmethod
    def _extract_required_posture(
            target_description: str,
            visual_requirements: dict | None) -> str | None:
        posture = SemanticTargetMatcherNode._extract_posture_from_text(
            target_description)
        if posture:
            return posture
        if not isinstance(visual_requirements, dict):
            return None
        for item in visual_requirements.get("conditions", []) or []:
            if not isinstance(item, dict):
                continue
            kind = str(item.get("kind", "") or "").strip().lower()
            cond = str(item.get("condition", "") or "").strip()
            if kind == "posture":
                posture = SemanticTargetMatcherNode._extract_posture_from_text(
                    cond)
                if posture:
                    return posture
        return None

    @staticmethod
    def _extract_candidate_posture(
            target_id: str, visual_facts: dict | None) -> str | None:
        if not isinstance(visual_facts, dict):
            return None
        for item in visual_facts.get("candidates", []) or []:
            if not isinstance(item, dict):
                continue
            cand_id = str(item.get("target_id", "") or "").strip().upper()
            if cand_id != str(target_id).strip().upper():
                continue
            posture = str(item.get("posture", "") or "").strip()
            return SemanticTargetMatcherNode._extract_posture_from_text(
                posture)
        return None

    @staticmethod
    def _extract_reference_constraints(
            target_description: str,
            visual_requirements: dict | None) -> list[dict]:
        result = []
        seen_names = set()
        if isinstance(visual_requirements, dict):
            for item in visual_requirements.get("reference_objects", []) or []:
                if not isinstance(item, dict):
                    continue
                name = str(item.get("name", "") or "").strip()
                relation = str(item.get("relation", "") or "").strip()
                if name and name not in seen_names:
                    seen_names.add(name)
                    result.append({"name": name, "relation": relation})
        desc = str(target_description or "")
        inferred_relation = ""
        compact = re.sub(r"\s+", "", desc)
        if any(word in compact for word in ("旁边", "附近", "靠近", "挨着")):
            inferred_relation = "near"
        aliases = [
            "花盆", "圆桶", "桶", "椅子", "桌子", "桌", "门", "白板",
            "包", "背包", "手机",
        ]
        for name in aliases:
            if name == "桶" and "圆桶" in seen_names:
                continue
            if name in compact and name not in seen_names:
                seen_names.add(name)
                result.append({"name": name, "relation": inferred_relation})
        return result

    @classmethod
    def _score_candidates_by_reference(
            cls, refs: list[dict], target_type: str | None,
            visual_facts: dict | None,
            candidates: list[dict]) -> tuple[dict, bool]:
        if not isinstance(visual_facts, dict):
            return {}, False
        same_type_ids = {
            str(item.get("target_id", "")).strip().upper()
            for item in candidates
            if _class_matches_target(
                item.get("target_class") or item.get("class"), target_type)
        }
        if not same_type_ids:
            return {}, False

        scores = {cid: 0 for cid in same_type_ids}
        found_reference = False
        alias_map = {
            "花盆": {"花盆", "盆栽", "绿植"},
            "圆桶": {"圆桶", "桶", "垃圾桶", "绿色桶", "绿桶"},
            "桶": {"桶", "圆桶", "垃圾桶", "绿色桶", "绿桶"},
            "椅子": {"椅子", "凳子", "座椅"},
            "桌子": {"桌子", "桌", "办公桌"},
            "桌": {"桌", "桌子", "办公桌"},
            "门": {"门"},
            "白板": {"白板", "黑板"},
            "包": {"包", "背包"},
            "背包": {"背包", "包"},
            "手机": {"手机"},
        }

        def matches_ref(name: str, ref_name: str) -> bool:
            text = str(name or "").strip()
            if not text:
                return False
            aliases = alias_map.get(ref_name, {ref_name})
            return any(alias in text or text in alias for alias in aliases)

        for cand in visual_facts.get("candidates", []) or []:
            if not isinstance(cand, dict):
                continue
            cid = str(cand.get("target_id", "") or "").strip().upper()
            if cid not in scores:
                continue
            nearby = cand.get("nearby_objects", []) or []
            if isinstance(nearby, str):
                nearby = [nearby]
            relations = cand.get("relations", []) or []
            for ref in refs:
                ref_name = ref["name"]
                for obj in nearby:
                    if matches_ref(str(obj), ref_name):
                        found_reference = True
                        scores[cid] += 4
                for rel in relations:
                    if not isinstance(rel, dict):
                        continue
                    to_name = str(rel.get("to", "") or "")
                    relation = str(rel.get("relation", "") or "")
                    evidence = str(rel.get("evidence", "") or "")
                    if (matches_ref(to_name, ref_name)
                            or matches_ref(relation, ref_name)
                            or matches_ref(evidence, ref_name)):
                        found_reference = True
                        scores[cid] += 5

        for obj in visual_facts.get("scene_objects", []) or []:
            if not isinstance(obj, dict):
                continue
            obj_name = str(obj.get("name", "") or "")
            near_ids = obj.get("near_candidates", []) or []
            if isinstance(near_ids, str):
                near_ids = [near_ids]
            near_ids = [
                str(item).strip().upper()
                for item in near_ids
                if str(item).strip().upper() in scores
            ]
            for ref in refs:
                ref_name = ref["name"]
                if not matches_ref(obj_name, ref_name):
                    continue
                found_reference = True
                for cid in near_ids:
                    scores[cid] += 6

        scores = {cid: score for cid, score in scores.items() if score > 0}
        return scores, found_reference

    @staticmethod
    def _extract_posture_from_text(text: str | None) -> str | None:
        compact = re.sub(r"\s+", "", str(text or "")).lower()
        if not compact:
            return None
        if any(word in compact for word in (
                "站着", "站立", "直立", "standing")):
            return "standing"
        if any(word in compact for word in (
                "坐着", "坐在", "坐姿", "倚坐", "sitting")):
            return "sitting"
        if any(word in compact for word in (
                "蹲着", "半蹲", "蹲姿", "squatting", "crouching")):
            return "squatting"
        if any(word in compact for word in (
                "躺着", "躺下", "平躺", "lying")):
            return "lying"
        return None

    @staticmethod
    def _posture_label(posture: str) -> str:
        labels = {
            "standing": "站着",
            "sitting": "坐着",
            "squatting": "蹲着",
            "lying": "躺着",
        }
        return labels.get(posture, posture)

    @staticmethod
    def _normalise_requirements(value) -> list[dict]:
        if not isinstance(value, list):
            return []
        result = []
        for item in value:
            if not isinstance(item, dict):
                continue
            req = str(item.get("requirement", "") or "").strip()
            status = str(item.get("status", "") or "").strip().lower()
            evidence = str(item.get("evidence", "") or "").strip()
            if status in ("yes", "true", "是", "满足"):
                status = "true"
            elif status in ("no", "false", "否", "不满足"):
                status = "false"
            elif status in ("unknown", "unclear", "不确定", "无法确认"):
                status = "uncertain"
            if not req or status not in {"true", "false", "uncertain"}:
                continue
            result.append({
                "requirement": req,
                "status": status,
                "evidence": evidence,
            })
        return result

    @staticmethod
    def _normalise_string_list(value) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            value = [value]
        if not isinstance(value, list):
            return []
        result = []
        for item in value:
            text = str(item or "").strip()
            if text and text.lower() not in {"null", "none", "无", "没有"}:
                result.append(text)
        return result

    @staticmethod
    def _normalise_candidate_id_list(value, candidates_by_id: dict) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            value = [value]
        if not isinstance(value, list):
            return []
        result = []
        for item in value:
            target_id = str(item or "").strip().upper()
            if target_id in {"", "NULL", "NONE", "无", "没有"}:
                continue
            if target_id not in candidates_by_id:
                continue
            if target_id not in result:
                result.append(target_id)
        return result

    @staticmethod
    def _normalise_referenced_objects(value) -> list[dict]:
        if not isinstance(value, list):
            return []
        result = []
        for item in value:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name", "") or "").strip()
            visible = item.get("visible")
            if isinstance(visible, str):
                visible = visible.strip().lower() in {
                    "true", "yes", "1", "是", "可见", "看到"}
            else:
                visible = bool(visible)
            location = item.get("location")
            relation = item.get("relation_to_target")
            evidence = str(item.get("evidence", "") or "").strip()
            result.append({
                "name": name,
                "visible": visible,
                "location": (
                    str(location).strip()
                    if location is not None else None),
                "relation_to_target": (
                    str(relation).strip()
                    if relation is not None else None),
                "evidence": evidence,
            })
        return result

    @staticmethod
    def _normalise_candidate_assessments(
            value, candidates_by_id: dict) -> list[dict]:
        if not isinstance(value, list):
            return []
        result = []
        for item in value:
            if not isinstance(item, dict):
                continue
            target_id = str(
                item.get("target_id") or item.get("person_id") or ""
            ).strip().upper()
            if target_id not in candidates_by_id:
                continue
            meets = item.get("meets")
            if isinstance(meets, str):
                meets = meets.strip().lower() in {
                    "true", "yes", "1", "是", "符合", "满足"}
            else:
                meets = bool(meets)
            reason = str(item.get("reason", "") or "").strip()
            result.append({
                "target_id": target_id,
                "meets": meets,
                "reason": reason,
            })
        return result

    def _publish_error(self, request_id: str, target_type: str | None,
                       target_description: str, user_request: str,
                       candidates: list[dict],
                       image_path: str, t0: float, reason: str):
        self.get_logger().error(reason)
        self._publish_result({
            "request_id": request_id,
            "match": False,
            "person_id": None,
            "target_type": target_type,
            "confidence": 0.0,
            "reason": reason,
            "requirements": [],
            "failed_requirements": [],
            "uncertain_requirements": [],
            "exclude_ids": [],
            "referenced_objects": [],
            "candidate_assessments": [],
            "target_description": target_description,
            "user_request": user_request,
            "candidates": candidates,
            "image_path": image_path,
            "duration_sec": round(time.time() - t0, 2),
            "fallback": True,
        })

    def _publish_result(self, result: dict):
        result.setdefault("target_id", None)
        result.setdefault("target_class", None)
        result.setdefault("target_type", None)
        result.setdefault("person_id", None)
        result.setdefault("answer", "是" if result.get("match") else "不是")
        result.setdefault("requirements", [])
        result.setdefault("failed_requirements", [])
        result.setdefault("uncertain_requirements", [])
        result.setdefault("exclude_ids", [])
        result.setdefault("referenced_objects", [])
        result.setdefault("candidate_assessments", [])
        msg = String()
        msg.data = json.dumps(result, ensure_ascii=False)
        self.result_pub.publish(msg)


PersonSemanticMatcherNode = SemanticTargetMatcherNode


def main(args=None, node_name: str = "person_semantic_matcher"):
    rclpy.init(args=args)
    node = SemanticTargetMatcherNode(node_name=node_name)
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
