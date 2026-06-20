#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WL100 巡检报告生成节点 (Mission Logger)

功能：
  订阅 demo 流程中的关键事件，按时间顺序记录，会话结束时
  生成"小白友好"的中文巡检报告（HTML + Markdown + JSON）。

订阅：
  /demo/narration_result            VLM 解说结果（核心数据来源）
  /navigate_to_pose/_action/status  Nav2 任务状态（识别到达事件）
  /odom                             累计行驶距离
  /demo/mission_start  (String)     用户触发：开始一次会话
  /demo/mission_end    (Empty)      用户触发：结束并生成报告

产物：
  ~/robot_ws/logs/巡检报告_<时间>/
    ├── 巡检报告.html        # 主报告（含样式、关键帧内嵌）
    ├── 巡检报告.md          # markdown 备份
    ├── 数据.json            # 结构化原始数据
    ├── 时间线.txt           # 纯文本时间线
    └── 关键帧/              # 拷贝并重命名的关键帧
"""

import json
import math
import os
import re
import shutil
import threading
import time
from datetime import datetime

import yaml
import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile, QoSReliabilityPolicy,
    QoSDurabilityPolicy, QoSHistoryPolicy)
from std_msgs.msg import String, Empty
from nav_msgs.msg import Odometry
from action_msgs.msg import GoalStatusArray, GoalStatus


# ════════════════════════════════════════════════════════
#  HTML 报告模板（暗色专业风）
# ════════════════════════════════════════════════════════
HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>{title}</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Microsoft YaHei",
                 "PingFang SC", "Hiragino Sans GB", sans-serif;
    background: #1a1c20; color: #e6e6e6;
    line-height: 1.7; padding: 24px;
  }}
  .container {{ max-width: 920px; margin: 0 auto; }}

  /* 顶部封面 */
  .cover {{
    background: linear-gradient(135deg, #4a90e2 0%, #6a4caf 100%);
    color: #fff; border-radius: 12px;
    padding: 36px 32px; margin-bottom: 24px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
  }}
  .cover h1 {{ font-size: 28px; margin-bottom: 16px; }}
  .cover-meta {{ font-size: 14px; opacity: 0.9; margin: 4px 0; }}
  .cover-status {{
    display: inline-block; margin-top: 12px;
    background: rgba(76, 175, 80, 0.3);
    border: 1px solid rgba(76, 175, 80, 0.6);
    padding: 6px 14px; border-radius: 4px; font-weight: bold;
  }}

  /* 概况卡片 */
  .summary {{
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 12px; margin-bottom: 24px;
  }}
  .stat-card {{
    background: #2b2d31; border-radius: 8px;
    padding: 18px; text-align: center;
    border: 1px solid #444;
  }}
  .stat-card .icon {{ font-size: 22px; margin-bottom: 4px; }}
  .stat-card .label {{ color: #888; font-size: 12px; margin-bottom: 6px; }}
  .stat-card .value {{ color: #fff; font-size: 22px; font-weight: bold; }}

  h2 {{
    color: #fff; font-size: 20px;
    margin: 32px 0 14px 0;
    padding-left: 12px; border-left: 4px solid #4a90e2;
  }}

  /* 时间线 */
  .timeline {{
    background: #2b2d31; border-radius: 8px;
    padding: 18px 22px; border: 1px solid #444;
  }}
  .timeline-item {{
    display: flex; padding: 6px 0;
    font-size: 13px; font-family: monospace;
  }}
  .timeline-time {{
    color: #888; margin-right: 14px; flex-shrink: 0;
  }}
  .timeline-text {{ color: #e6e6e6; }}

  /* 航点卡片 */
  .station {{
    background: #2b2d31; border-radius: 10px;
    padding: 24px; margin-bottom: 18px;
    border: 1px solid #444;
  }}
  .station.has-person {{
    border: 2px solid #e74c3c;
    box-shadow: 0 0 20px rgba(231, 76, 60, 0.3);
  }}
  .station h3 {{ color: #fff; font-size: 18px; margin-bottom: 12px; }}
  .station h3 .badge {{
    display: inline-block; background: #4a90e2;
    color: #fff; padding: 2px 10px;
    border-radius: 12px; font-size: 12px; margin-left: 8px;
    font-weight: normal;
  }}
  .station h3 .badge-person {{ background: #e74c3c; }}

  .station-meta {{
    display: grid; grid-template-columns: repeat(2, 1fr);
    gap: 8px; margin-bottom: 14px;
    color: #aaa; font-size: 12px;
  }}
  .station-meta span {{ display: block; }}
  .station-meta strong {{ color: #c0c4cc; }}

  .station img {{
    max-width: 100%; border-radius: 6px;
    margin: 12px 0; border: 1px solid #555;
  }}
  .field {{ margin: 10px 0; }}
  .field-label {{
    color: #c0c4cc; font-weight: bold;
    font-size: 13px; margin-bottom: 4px;
  }}
  .field-value {{ color: #e6e6e6; font-size: 14px; }}
  .field-value ul {{ padding-left: 20px; }}

  .person-detected {{
    background: rgba(231, 76, 60, 0.15);
    border-left: 4px solid #e74c3c;
    padding: 12px 16px; border-radius: 4px;
    color: #ffb3b3;
  }}
  .person-not-detected {{
    background: rgba(136, 136, 136, 0.1);
    border-left: 4px solid #888;
    padding: 10px 14px; border-radius: 4px;
    color: #aaa;
  }}

  /* 综合建议 */
  .advice-block {{
    background: #2b2d31; border-radius: 8px;
    padding: 22px 26px; border-left: 4px solid #d4a017;
  }}
  .advice-block ul {{ padding-left: 20px; margin-top: 8px; }}
  .advice-block li {{ margin: 4px 0; }}

  /* 页脚 */
  .footer {{
    margin-top: 36px; padding-top: 18px;
    border-top: 1px solid #333;
    text-align: center; color: #666; font-size: 12px;
  }}
</style>
</head>
<body>
<div class="container">

  <div class="cover">
    <h1>🤖 WL100 智能巡检报告</h1>
    <div class="cover-meta">📋 任务编号：{mission_id}</div>
    <div class="cover-meta">🕒 开始时间：{start_time}</div>
    <div class="cover-meta">🕔 结束时间：{end_time}</div>
    <div class="cover-meta">⏱️ 总耗时：{duration_human}</div>
    <div class="cover-status">{status_text}</div>
  </div>

  <h2>📊 巡检概况</h2>
  <div class="summary">
    <div class="stat-card">
      <div class="icon">🛰️</div>
      <div class="label">巡检航点</div>
      <div class="value">{station_count}</div>
    </div>
    <div class="stat-card">
      <div class="icon">🚶</div>
      <div class="label">发现人员</div>
      <div class="value">{person_count}</div>
    </div>
    <div class="stat-card">
      <div class="icon">⚠️</div>
      <div class="label">异常事件</div>
      <div class="value">{anomaly_count}</div>
    </div>
    <div class="stat-card">
      <div class="icon">🚀</div>
      <div class="label">导航成功率</div>
      <div class="value">{nav_success_rate}</div>
    </div>
    <div class="stat-card">
      <div class="icon">📸</div>
      <div class="label">关键帧</div>
      <div class="value">{keyframe_count}</div>
    </div>
    <div class="stat-card">
      <div class="icon">📏</div>
      <div class="label">总行驶距离</div>
      <div class="value">{total_distance_str}</div>
    </div>
  </div>

  <h2>📅 任务时间线</h2>
  <div class="timeline">
    {timeline_html}
  </div>

  <h2>📍 巡检详情</h2>
  {stations_html}

  <h2>💡 综合建议</h2>
  <div class="advice-block">
    {advice_html}
  </div>

  <div class="footer">
    本报告由 WL100 巡检机器人自动生成<br>
    Robot Operating System: ROS2 Humble &nbsp;·&nbsp;
    VLM Backend: Cosmos Reason2 2B
  </div>

</div>
</body>
</html>
"""


# ════════════════════════════════════════════════════════
#                       节点
# ════════════════════════════════════════════════════════
class MissionLoggerNode(Node):
    def __init__(self):
        super().__init__("mission_logger")

        # ── 参数 ──
        self.declare_parameter("narration_topic", "/demo/narration_result")
        self.declare_parameter(
            "nav_status_topic", "/navigate_to_pose/_action/status")
        self.declare_parameter("odom_topic", "/odom")
        self.declare_parameter("mission_start_topic", "/demo/mission_start")
        self.declare_parameter("mission_end_topic", "/demo/mission_end")
        self.declare_parameter("waypoints_yaml",
            "/home/nvidia/robot_ws/src/wl100_demo/config/waypoints.yaml")
        self.declare_parameter("output_root",
            "/home/nvidia/robot_ws/logs")
        self.declare_parameter("auto_start_on_launch", True)

        narration_topic = self.get_parameter("narration_topic").value
        nav_status_topic = self.get_parameter("nav_status_topic").value
        odom_topic = self.get_parameter("odom_topic").value
        mission_start_topic = self.get_parameter("mission_start_topic").value
        mission_end_topic = self.get_parameter("mission_end_topic").value
        self.waypoints_yaml = self.get_parameter("waypoints_yaml").value
        self.output_root = os.path.abspath(
            os.path.expanduser(self.get_parameter("output_root").value))
        auto_start = bool(self.get_parameter("auto_start_on_launch").value)

        os.makedirs(self.output_root, exist_ok=True)

        # ── 状态 ──
        self.lock = threading.Lock()
        self.session_active = False
        self.session_dir = ""
        self.session_id = ""
        self.session_title = ""
        self.session_start_t = 0.0
        self.session_end_t = 0.0
        self.events = []  # [(timestamp, kind, payload_dict)]
        self.narrations = []  # 完整 narration_result 列表
        self.nav_status_history = []  # 简化的导航状态变化
        self.last_nav_status = None
        self.cumulative_distance = 0.0
        self.last_odom_pos = None
        self.waypoints = self._load_waypoints()

        # ── 订阅 ──
        self.create_subscription(
            String, narration_topic, self._narration_cb, 10)
        self.create_subscription(
            GoalStatusArray, nav_status_topic, self._nav_status_cb, 10)
        self.create_subscription(
            Odometry, odom_topic, self._odom_cb, 10)
        self.create_subscription(
            String, mission_start_topic, self._mission_start_cb, 10)
        self.create_subscription(
            Empty, mission_end_topic, self._mission_end_cb, 10)

        # ── 发布：当前会话目录（latched，让 scene_narrator 后启动也能拿到）──
        latched_qos = QoSProfile(
            depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            history=QoSHistoryPolicy.KEEP_LAST)
        self.session_dir_pub = self.create_publisher(
            String, "/demo/log_dir", latched_qos)

        self.get_logger().info("════════════════════════════════════════")
        self.get_logger().info("📒 WL100 巡检报告生成节点已启动")
        self.get_logger().info(f"  输出根目录:  {self.output_root}")
        self.get_logger().info(f"  解说话题:    {narration_topic}")
        self.get_logger().info(f"  开始话题:    {mission_start_topic}")
        self.get_logger().info(f"  结束话题:    {mission_end_topic}")
        self.get_logger().info(f"  已加载航点:  {len(self.waypoints)} 个")
        self.get_logger().info("════════════════════════════════════════")
        self.get_logger().info("💡 用法：")
        self.get_logger().info(
            f"  ros2 topic pub --once {mission_start_topic} "
            f"std_msgs/msg/String \"{{data: '巡检 A B C 观测点'}}\"")
        self.get_logger().info(
            f"  ros2 topic pub --once {mission_end_topic} "
            f"std_msgs/msg/Empty \"{{}}\"")

        if auto_start:
            self._start_session("自动启动")

    # ────────────────────────────────────────────
    #  YAML 加载
    # ────────────────────────────────────────────
    def _load_waypoints(self):
        try:
            with open(self.waypoints_yaml, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            wps = data.get("waypoints") or {}
            return {str(k): {
                "x": float(v.get("x", 0.0)),
                "y": float(v.get("y", 0.0)),
                "yaw": float(v.get("yaw", 0.0)),
            } for k, v in wps.items()}
        except Exception as e:
            self.get_logger().warn(f"加载航点 yaml 失败: {e}")
            return {}

    # ────────────────────────────────────────────
    #  会话管理
    # ────────────────────────────────────────────
    def _start_session(self, title: str):
        with self.lock:
            if self.session_active:
                self.get_logger().warn("已有会话进行中，忽略 mission_start")
                return
            now = datetime.now()
            self.session_id = "WL100-" + now.strftime("%Y%m%d-%H%M%S")
            self.session_title = title
            self.session_start_t = time.time()
            # 中文星期映射
            weekday_zh = ["一", "二", "三", "四", "五", "六", "日"][now.weekday()]
            human_dir = (
                f"巡检报告_{now.year}年{now.month:02d}月{now.day:02d}日_"
                f"周{weekday_zh}_{now.hour:02d}点{now.minute:02d}分")
            self.session_dir = os.path.join(self.output_root, human_dir)
            os.makedirs(os.path.join(self.session_dir, "关键帧"),
                exist_ok=True)
            self.events = []
            self.narrations = []
            self.nav_status_history = []
            self.last_nav_status = None
            self.cumulative_distance = 0.0
            self.last_odom_pos = None
            self.session_active = True

        self._add_event("session_start",
            {"title": title, "id": self.session_id})
        self.get_logger().info(
            f"🟢 新会话已开启: {self.session_id}")
        self.get_logger().info(f"   输出目录: {self.session_dir}")

        # 关键帧目标目录（让 scene_narrator 直接落到这里）
        self._publish_session_dir(
            os.path.join(self.session_dir, "关键帧"))

    def _end_session(self):
        with self.lock:
            if not self.session_active:
                self.get_logger().warn("当前无进行中的会话，忽略 mission_end")
                return
            self.session_end_t = time.time()
            self.session_active = False
            n_narr = len(self.narrations)
            n_nav = len(self.nav_status_history)

        # ★ 空会话保护：没有任何解说 + 没有任何导航 → 不生成空报告
        #   （防止连点两次"生成报告"产出一份垃圾空报告）
        if n_narr == 0 and n_nav == 0:
            self.get_logger().info(
                "⊘ 本会话无任何巡检数据（0 解说 0 导航），跳过报告生成")
            # 删掉刚建的空会话目录（只删空的，非空不动）
            try:
                kf_dir = os.path.join(self.session_dir, "关键帧")
                if os.path.isdir(kf_dir) and not os.listdir(kf_dir):
                    os.rmdir(kf_dir)
                if os.path.isdir(self.session_dir) and not os.listdir(
                        self.session_dir):
                    os.rmdir(self.session_dir)
            except Exception:
                pass
            self._publish_session_dir(self.output_root)
            return

        self._add_event("session_end", {})
        self.get_logger().info("🔴 会话已结束，正在生成报告...")
        try:
            self._generate_report()
            self.get_logger().info(
                f"✅ 报告已生成: {self.session_dir}/巡检报告.html")
        except Exception as e:
            self.get_logger().error(f"生成报告失败: {e}")
            import traceback
            self.get_logger().error(traceback.format_exc())

        # 会话结束 → 通知 scene_narrator 切回根目录
        self._publish_session_dir(self.output_root)

    def _publish_session_dir(self, path: str):
        """发布关键帧目标目录到 latched 话题 /demo/log_dir。
        scene_narrator 订阅后会把 narrate_*.jpg 直接落到这里。"""
        m = String()
        m.data = path
        try:
            self.session_dir_pub.publish(m)
            self.get_logger().info(f"📡 通知 scene_narrator 落盘到: {path}")
        except Exception as e:
            self.get_logger().warn(f"发布 session_dir 失败: {e}")

    # ────────────────────────────────────────────
    #  事件追加
    # ────────────────────────────────────────────
    def _add_event(self, kind: str, payload: dict):
        with self.lock:
            self.events.append({
                "ts": time.time(),
                "kind": kind,
                **payload,
            })

    # ────────────────────────────────────────────
    #  ROS 回调
    # ────────────────────────────────────────────
    def _mission_start_cb(self, msg: String):
        title = (msg.data or "").strip() or "巡检任务"
        self._start_session(title)

    def _mission_end_cb(self, msg: Empty):
        self._end_session()
        # 自动开新会话，继续记录后续事件
        self._start_session("续接（上一份报告已生成）")

    def _narration_cb(self, msg: String):
        if not self.session_active:
            return
        try:
            data = json.loads(msg.data)
        except json.JSONDecodeError:
            self.get_logger().warn("narration_result JSON 解析失败")
            return
        with self.lock:
            self.narrations.append(data)
        self._add_event("narration", data)
        self.get_logger().info(
            f"📝 已记录 VLM 解说: mode={data.get('trigger', {}).get('mode')}, "
            f"wp={data.get('trigger', {}).get('waypoint')}, "
            f"has_person={data.get('has_person')}")

    def _nav_status_cb(self, msg: GoalStatusArray):
        if not self.session_active:
            return
        # 我们只关心 status 变化（避免每秒打很多）
        if not msg.status_list:
            return
        last = msg.status_list[-1]
        st = last.status
        if st == self.last_nav_status:
            return
        self.last_nav_status = st
        kind_map = {
            GoalStatus.STATUS_ACCEPTED: "nav_accepted",
            GoalStatus.STATUS_EXECUTING: "nav_executing",
            GoalStatus.STATUS_SUCCEEDED: "nav_succeeded",
            GoalStatus.STATUS_CANCELED: "nav_canceled",
            GoalStatus.STATUS_ABORTED: "nav_aborted",
        }
        kind = kind_map.get(st, f"nav_status_{st}")
        with self.lock:
            self.nav_status_history.append((time.time(), kind))
        self._add_event(kind, {})

    def _odom_cb(self, msg: Odometry):
        if not self.session_active:
            return
        p = msg.pose.pose.position
        if self.last_odom_pos is None:
            self.last_odom_pos = (p.x, p.y)
            return
        dx = p.x - self.last_odom_pos[0]
        dy = p.y - self.last_odom_pos[1]
        d = math.hypot(dx, dy)
        # 过滤微小抖动（< 0.5cm 不算）
        if d <= 0.005:
            return
        # ★ 过滤定位跳变：单帧位移 > 0.5m 几乎不可能是真实移动
        #   （车 vx_max=0.2m/s，正常单帧远 < 0.5m），是 HDL/LIO 跳变，
        #   不累加假距离，但更新基准位置避免后续连锁误差
        if d > 0.5:
            self.last_odom_pos = (p.x, p.y)
            return
        self.cumulative_distance += d
        self.last_odom_pos = (p.x, p.y)

    # ────────────────────────────────────────────
    #  报告生成
    # ────────────────────────────────────────────
    def _generate_report(self):
        # 拷贝关键帧并重命名
        keyframes = []
        copied_src = set()    # 已成功处理的源文件（最后统一清理 narrate_*.jpg）
        for i, narr in enumerate(self.narrations, 1):
            src_path = narr.get("image_path") or ""
            if not src_path or not os.path.isfile(src_path):
                continue
            mode = narr.get("trigger", {}).get("mode", "unknown")
            wp = narr.get("trigger", {}).get("waypoint") or "未知"
            wp_safe = re.sub(r"[^\w\u4e00-\u9fff\-]", "_", wp)
            mode_zh = {
                "arrive": "到达",
                "find_person": "接近人员",
                "no_person_advice": "未见人员",
                "patrol": "巡检",
            }.get(mode, mode)
            ext = os.path.splitext(src_path)[1] or ".jpg"
            new_name = f"{i:02d}_{mode_zh}_{wp_safe}{ext}"
            dst_path = os.path.join(self.session_dir, "关键帧", new_name)
            try:
                # ★ 统一用 copy（不 move）：
                #   同一张源图可能被多条 narration 引用（如 arrive + find_person
                #   抓到同名帧），move 会让第二次找不到源文件丢图。
                #   copy 保证每条都能成功，源图最后统一清理。
                if os.path.abspath(src_path) != os.path.abspath(dst_path):
                    shutil.copy2(src_path, dst_path)
                    copied_src.add(os.path.abspath(src_path))
                keyframes.append((new_name, narr))
            except Exception as e:
                self.get_logger().warn(f"拷贝关键帧失败 {src_path}: {e}")

        # 清理原始 narrate_*.jpg（已复制成 NN_模式_观测点.jpg，原名图删掉省空间）
        for src in copied_src:
            try:
                base = os.path.basename(src)
                # 只删 scene_narrator 产出的 narrate_ 前缀临时图，避免误删
                if base.startswith("narrate_") and os.path.isfile(src):
                    os.remove(src)
            except Exception:
                pass

        # 1. 数据.json
        json_path = os.path.join(self.session_dir, "数据.json")
        struct_data = {
            "mission_id": self.session_id,
            "title": self.session_title,
            "start_time": self._fmt_time(self.session_start_t),
            "end_time": self._fmt_time(self.session_end_t),
            "duration_sec": round(self.session_end_t - self.session_start_t, 1),
            "total_distance_m": round(self.cumulative_distance, 2),
            "waypoints_loaded": self.waypoints,
            "narrations": self.narrations,
            "events": self.events,
            "nav_status_history": [
                {"ts": ts, "status": k}
                for ts, k in self.nav_status_history
            ],
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(struct_data, f, ensure_ascii=False, indent=2)

        # 2. 时间线.txt
        timeline_path = os.path.join(self.session_dir, "时间线.txt")
        with open(timeline_path, "w", encoding="utf-8") as f:
            f.write(f"=== {self.session_title} 时间线 ===\n")
            f.write(f"会话编号: {self.session_id}\n\n")
            for ev in self.events:
                t_str = datetime.fromtimestamp(ev["ts"]).strftime("%H:%M:%S")
                f.write(f"[{t_str}] {self._event_to_text(ev)}\n")

        # 3. Markdown 报告
        md_text = self._build_markdown(keyframes)
        with open(os.path.join(self.session_dir, "巡检报告.md"), "w",
                  encoding="utf-8") as f:
            f.write(md_text)

        # 4. HTML 报告
        html_text = self._build_html(keyframes)
        with open(os.path.join(self.session_dir, "巡检报告.html"), "w",
                  encoding="utf-8") as f:
            f.write(html_text)

        self.get_logger().info(
            f"📦 共 {len(keyframes)} 张关键帧 / "
            f"{len(self.narrations)} 次 VLM 解说 / "
            f"行驶 {self.cumulative_distance:.2f} m")

    # ────────────────────────────────────────────
    #  报告构建工具
    # ────────────────────────────────────────────
    def _fmt_time(self, t: float) -> str:
        return datetime.fromtimestamp(t).strftime("%Y-%m-%d %H:%M:%S")

    def _fmt_duration(self, sec: float) -> str:
        m, s = divmod(int(sec), 60)
        h, m = divmod(m, 60)
        if h:
            return f"{h} 小时 {m} 分 {s} 秒"
        if m:
            return f"{m} 分 {s} 秒"
        return f"{s} 秒"

    def _event_to_text(self, ev) -> str:
        k = ev["kind"]
        if k == "session_start":
            return f"🟢 任务开启 [{ev.get('title', '')}]"
        if k == "session_end":
            return "🔴 任务结束"
        if k == "narration":
            mode = ev.get("trigger", {}).get("mode", "?")
            wp = ev.get("trigger", {}).get("waypoint") or ""
            d = ev.get("description", "")
            return f"🎙️ VLM[{mode}@{wp}] {d}"
        if k == "nav_accepted":
            return "🛰️ 导航请求已接受"
        if k == "nav_executing":
            return "🚀 导航执行中"
        if k == "nav_succeeded":
            return "📍 已到达目标航点"
        if k == "nav_canceled":
            return "⊗ 导航被取消"
        if k == "nav_aborted":
            return "✕ 导航中止"
        return k

    def _compute_stats(self):
        nav_total = sum(1 for _, k in self.nav_status_history
                        if k == "nav_succeeded") + \
                    sum(1 for _, k in self.nav_status_history
                        if k in ("nav_canceled", "nav_aborted"))
        nav_success = sum(1 for _, k in self.nav_status_history
                          if k == "nav_succeeded")
        if nav_total == 0:
            nav_rate = "—"
        else:
            nav_rate = f"{int(nav_success * 100 / nav_total)}%"

        person_count = sum(1 for n in self.narrations if n.get("has_person"))

        # 异常 = nav_aborted + fallback narrations
        anomaly = sum(1 for _, k in self.nav_status_history
                      if k in ("nav_aborted",))
        anomaly += sum(1 for n in self.narrations if n.get("fallback"))

        # 站点数：以 narrations 中 mode=arrive/patrol 的不同 waypoint 计数
        stations = set()
        for n in self.narrations:
            mode = n.get("trigger", {}).get("mode")
            wp = n.get("trigger", {}).get("waypoint")
            if mode in ("arrive", "patrol") and wp:
                stations.add(wp)
        station_count = len(stations) if stations else len(self.narrations)

        return {
            "station_count": station_count,
            "person_count": person_count,
            "anomaly_count": anomaly,
            "nav_success_rate": nav_rate,
        }

    def _build_advice(self) -> list:
        """根据 narrations 生成"综合建议"段落（小白友好）"""
        advice = []

        # 整体状态
        succeeded = sum(1 for _, k in self.nav_status_history
                        if k == "nav_succeeded")
        aborted = sum(1 for _, k in self.nav_status_history
                      if k in ("nav_aborted", "nav_canceled"))
        if aborted == 0 and self.narrations:
            advice.append("✅ 整体情况：所有航点全部到达，无异常事件。")
        elif aborted > 0:
            advice.append(
                f"⚠️ 整体情况：{succeeded} 次导航成功，"
                f"{aborted} 次导航中断，请关注被中断的航点。")

        # 人员情况
        person_wps = []
        empty_wps = []
        for n in self.narrations:
            mode = n.get("trigger", {}).get("mode")
            wp = n.get("trigger", {}).get("waypoint") or ""
            if mode not in ("arrive", "patrol"):
                continue
            if n.get("has_person"):
                person_wps.append(wp)
            else:
                empty_wps.append(wp)
        if person_wps:
            advice.append(
                "👥 人员状态：" + "、".join(set(person_wps)) +
                " 观测点有人在岗。")
        if empty_wps:
            advice.append(
                "🚪 无人观测点：" + "、".join(set(empty_wps)) +
                " 观测点无人。")

        # 关注点（线缆 / 插排 / 纸箱过多等）
        attention = []
        for n in self.narrations:
            wp = n.get("trigger", {}).get("waypoint") or ""
            ko = n.get("key_objects") or []
            for risky in ("线缆", "电线", "插排"):
                if risky in ko and wp:
                    attention.append(
                        f"{wp} 观测点发现 {risky}，建议关注通道整洁")
        if attention:
            advice.append("⚠️ 关注点：")
            advice.extend(["  • " + a for a in dict.fromkeys(attention)])

        # VLM 解说成功率
        total_vlm = len(self.narrations)
        fallback_cnt = sum(1 for n in self.narrations if n.get("fallback"))
        if total_vlm:
            ok_rate = int((total_vlm - fallback_cnt) * 100 / total_vlm)
            advice.append(
                f"📈 性能：VLM 解说成功率 {ok_rate}%，"
                f"总行驶距离约 {self.cumulative_distance:.1f} 米。")

        if not advice:
            advice.append("（暂无可总结内容）")
        return advice

    # ────────────────────────────────────────────
    #  Markdown 报告
    # ────────────────────────────────────────────
    def _build_markdown(self, keyframes) -> str:
        stats = self._compute_stats()
        duration = self.session_end_t - self.session_start_t

        lines = []
        lines.append(f"# 🤖 WL100 智能巡检报告\n")
        lines.append(f"**任务编号**：{self.session_id}\n")
        lines.append(f"**开始时间**：{self._fmt_time(self.session_start_t)}\n")
        lines.append(f"**结束时间**：{self._fmt_time(self.session_end_t)}\n")
        lines.append(f"**总耗时**：{self._fmt_duration(duration)}\n")
        lines.append(f"**任务标题**：{self.session_title}\n")
        lines.append("")
        lines.append("---\n")
        lines.append("## 📊 巡检概况\n")
        lines.append("| 项目 | 数值 |")
        lines.append("|---|---|")
        lines.append(f"| 巡检航点 | {stats['station_count']} 个 |")
        lines.append(f"| 发现人员 | {stats['person_count']} 处 |")
        lines.append(f"| 异常事件 | {stats['anomaly_count']} |")
        lines.append(f"| 导航成功率 | {stats['nav_success_rate']} |")
        lines.append(f"| 关键帧数量 | {len(keyframes)} |")
        lines.append(
            f"| 总行驶距离 | {self.cumulative_distance:.2f} 米 |")
        lines.append("")
        lines.append("---\n")

        # 时间线
        lines.append("## 📅 任务时间线\n")
        for ev in self.events:
            t_str = datetime.fromtimestamp(ev["ts"]).strftime("%H:%M:%S")
            lines.append(f"- `{t_str}` — {self._event_to_text(ev)}")
        lines.append("")
        lines.append("---\n")

        # 巡检详情
        lines.append("## 📍 巡检详情\n")
        for idx, (filename, narr) in enumerate(keyframes, 1):
            mode = narr.get("trigger", {}).get("mode", "?")
            wp = narr.get("trigger", {}).get("waypoint") or "未知"
            mode_zh = {
                "arrive": "到达",
                "find_person": "接近人员",
                "no_person_advice": "未见人员建议",
                "patrol": "巡检",
            }.get(mode, mode)

            wp_meta = self.waypoints.get(wp, {})
            wp_xy = (f"({wp_meta.get('x', 0.0):.2f}, "
                     f"{wp_meta.get('y', 0.0):.2f})") \
                    if wp_meta else "(未配置)"

            person_flag = "✅ 发现人员" if narr.get("has_person") \
                else "未发现人员"

            lines.append(f"### 第 {idx} 站：{wp}（{mode_zh}）\n")
            lines.append(f"![](关键帧/{filename})\n")
            lines.append(f"- **航点坐标**：{wp_xy}")
            lines.append(f"- **模式**：{mode_zh}")
            lines.append(f"- **现场情况**：{narr.get('description', '')}")
            lines.append(f"- **人员检测**：{person_flag}")
            if narr.get("has_person") and narr.get("person_position_hint"):
                lines.append(
                    f"- **人员位置**：{narr.get('person_position_hint')}")
                lines.append(
                    f"- **人员数量**：{narr.get('person_count', 1)}")
            ko = narr.get("key_objects") or []
            if ko:
                lines.append(f"- **关键物品**：{ '、'.join(ko) }")
            if narr.get("advice"):
                lines.append(f"- **AI 建议**：{narr.get('advice')}")
            lines.append(
                f"- **VLM 推理耗时**：{narr.get('duration_sec', 0)} 秒")
            lines.append("")

        lines.append("---\n")
        lines.append("## 💡 综合建议\n")
        for line in self._build_advice():
            lines.append(line)
        lines.append("")
        lines.append("---\n")
        lines.append(
            "*本报告由 WL100 巡检机器人自动生成 · "
            "ROS2 Humble · Cosmos Reason2 2B*")

        return "\n".join(lines)

    # ────────────────────────────────────────────
    #  HTML 报告
    # ────────────────────────────────────────────
    def _build_html(self, keyframes) -> str:
        stats = self._compute_stats()
        duration = self.session_end_t - self.session_start_t

        # 时间线 HTML
        timeline_items = []
        for ev in self.events:
            t_str = datetime.fromtimestamp(ev["ts"]).strftime("%H:%M:%S")
            text = self._event_to_text(ev)
            timeline_items.append(
                f'<div class="timeline-item">'
                f'<span class="timeline-time">{t_str}</span>'
                f'<span class="timeline-text">{self._html_escape(text)}</span>'
                f'</div>')
        timeline_html = "\n".join(timeline_items)

        # 站点 HTML
        stations_html = []
        for idx, (filename, narr) in enumerate(keyframes, 1):
            mode = narr.get("trigger", {}).get("mode", "?")
            wp = narr.get("trigger", {}).get("waypoint") or "未知"
            mode_zh = {
                "arrive": "到达",
                "find_person": "接近人员",
                "no_person_advice": "未见人员建议",
                "patrol": "巡检",
            }.get(mode, mode)
            has_p = bool(narr.get("has_person"))

            wp_meta = self.waypoints.get(wp, {})
            wp_xy = (f"({wp_meta.get('x', 0.0):.2f}, "
                     f"{wp_meta.get('y', 0.0):.2f})") \
                    if wp_meta else "(未配置)"

            badge_cls = "badge badge-person" if has_p else "badge"
            badge_text = "🚨 发现人员" if has_p else mode_zh
            station_cls = "station has-person" if has_p else "station"

            ko = narr.get("key_objects") or []
            ko_html = ("<ul>" + "".join(
                f"<li>{self._html_escape(o)}</li>" for o in ko)
                + "</ul>") if ko else "<em>无</em>"

            person_html = ""
            if has_p:
                pos = narr.get("person_position_hint") or "—"
                cnt = narr.get("person_count", 1)
                person_html = (
                    f'<div class="person-detected">'
                    f'<strong>🚨 检测到 {cnt} 人</strong><br>'
                    f'位置：{self._html_escape(pos)}'
                    f'</div>')
            else:
                person_html = (
                    '<div class="person-not-detected">'
                    '未在视野中发现人员</div>')

            advice_html = ""
            if narr.get("advice"):
                advice_html = (
                    '<div class="field"><div class="field-label">AI 建议</div>'
                    f'<div class="field-value">'
                    f'{self._html_escape(narr.get("advice"))}</div></div>')

            stations_html.append(f"""
<div class="{station_cls}">
  <h3>第 {idx} 站：{self._html_escape(wp)}
    <span class="{badge_cls}">{badge_text}</span></h3>
  <div class="station-meta">
    <span><strong>航点坐标</strong>：{wp_xy}</span>
    <span><strong>模式</strong>：{mode_zh}</span>
    <span><strong>VLM 耗时</strong>：{narr.get("duration_sec", 0)} 秒</span>
    <span><strong>fallback</strong>：{"是" if narr.get("fallback") else "否"}</span>
  </div>
  <img src="关键帧/{self._html_escape(filename)}" alt="现场画面">
  <div class="field">
    <div class="field-label">现场情况</div>
    <div class="field-value">{self._html_escape(narr.get("description", ""))}</div>
  </div>
  <div class="field">
    <div class="field-label">人员检测</div>
    <div class="field-value">{person_html}</div>
  </div>
  <div class="field">
    <div class="field-label">关键物品</div>
    <div class="field-value">{ko_html}</div>
  </div>
  {advice_html}
</div>
""")
        stations_html_str = "\n".join(stations_html)

        # 综合建议 HTML
        advice_lines = self._build_advice()
        advice_html = "<ul>" + "".join(
            f"<li>{self._html_escape(line)}</li>" for line in advice_lines
        ) + "</ul>"

        # 状态文本
        if stats["anomaly_count"] == 0:
            status_text = "✅ 任务成功完成"
        else:
            status_text = f"⚠️ 任务完成，{stats['anomaly_count']} 处异常"

        # 总距离格式化
        d = self.cumulative_distance
        if d >= 1000:
            total_distance_str = f"{d/1000:.2f} 公里"
        else:
            total_distance_str = f"{d:.2f} 米"

        # 用逐个 replace 替换占位符（避免内容里的 {} 触发 .format 崩溃）
        # ⚠️ 不能用 HTML_TEMPLATE.format()，因为 description 等动态内容可能
        #    含 { } 字符（VLM 输出 JSON 片段/表情），会让 .format() 抛 KeyError
        _subs = {
            "{title}": f"WL100 巡检报告 {self.session_id}",
            "{mission_id}": str(self.session_id),
            "{start_time}": self._fmt_time(self.session_start_t),
            "{end_time}": self._fmt_time(self.session_end_t),
            "{duration_human}": self._fmt_duration(duration),
            "{status_text}": status_text,
            "{station_count}": str(stats["station_count"]),
            "{person_count}": str(stats["person_count"]),
            "{anomaly_count}": str(stats["anomaly_count"]),
            "{nav_success_rate}": str(stats["nav_success_rate"]),
            "{keyframe_count}": str(len(keyframes)),
            "{total_distance_str}": total_distance_str,
            "{timeline_html}": timeline_html,
            "{stations_html}": stations_html_str,
            "{advice_html}": advice_html,
        }
        # 先把 CSS 里的转义双花括号 {{ }} 还原成单花括号
        html = HTML_TEMPLATE.replace("{{", "{").replace("}}", "}")
        for k, v in _subs.items():
            html = html.replace(k, v)
        return html

    @staticmethod
    def _html_escape(s) -> str:
        if s is None:
            return ""
        s = str(s)
        return (s.replace("&", "&amp;")
                 .replace("<", "&lt;")
                 .replace(">", "&gt;")
                 .replace('"', "&quot;")
                 .replace("{", "&#123;")
                 .replace("}", "&#125;"))


def main(args=None):
    rclpy.init(args=args)
    node = MissionLoggerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Ctrl+C 时如果还有会话，自动结束并生成报告
        if node.session_active:
            node.get_logger().info(
                "📕 检测到 Ctrl+C，自动结束当前会话并生成报告...")
            try:
                node._end_session()
            except Exception:
                pass
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
