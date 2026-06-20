#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WL100 自然语言指令解析节点 (NL Parser)

外部接口保持不变：
  订阅 /demo/nl_command、/demo/narration_result、/demo/tts/done、/demo/cancel
  发布 /demo/nl_reply、/demo/run_stage、/demo/cancel、/demo/mission_end

内部链路：
  VLM Planner -> IntentValidator -> Decision -> TaskExecutor

关键约束：
  VLM 只负责理解自然语言并输出语义 steps；
  本地 validator 负责缺槽位追问、白名单校验和 stage 映射；
  executor 是唯一执行出口，必须先播回复，TTS 完成后才下发任务。
"""

import json
import re
import threading
import time
from collections import deque

import rclpy
from rclpy.node import Node
from std_msgs.msg import Empty, String

from wl100_demo.conversation_memory import ConversationMemory
from wl100_demo.intent_validator import IntentValidator
from wl100_demo.nl_prompt import build_system_prompt, today_human
from wl100_demo.nlu_schema import Decision, StageAction, format_stage_actions
from wl100_demo.skill_mapper import build_action_reply
from wl100_demo.task_segmenter import segment_tasks
from wl100_demo.task_executor import TaskExecutor
from wl100_demo.vlm_client import VlmClient, VlmClientError
from wl100_demo.vlm_planner import VlmPlanner


CANCEL_PATTERNS = [
    r"^取消$", r"^停止$", r"^停$", r"^停下$",
    r"^cancel$", r"^stop$", r"^abort$",
    r"别走了", r"停一下",
]


class NlParserNode(Node):
    def __init__(self):
        super().__init__("nl_parser")

        self.declare_parameter("nl_command_topic", "/demo/nl_command")
        self.declare_parameter("run_stage_topic", "/demo/run_stage")
        self.declare_parameter("cancel_topic", "/demo/cancel")
        self.declare_parameter("reply_topic", "/demo/nl_reply")
        self.declare_parameter(
            "vlm_api_url", "http://localhost:8080/v1/chat/completions")
        self.declare_parameter("vlm_timeout", 20.0)
        self.declare_parameter("vlm_temperature", 0.4)
        self.declare_parameter("vlm_max_tokens", 2048)
        self.declare_parameter("history_max", 20)
        self.declare_parameter("use_local_fast_path", False)
        self.declare_parameter("vlm_observe_only", True)
        self.declare_parameter("tts_done_topic", "/demo/tts/done")
        self.declare_parameter("wait_for_tts_before_action", True)
        self.declare_parameter("tts_wait_timeout_sec", 60.0)

        nl_topic = self.get_parameter("nl_command_topic").value
        stage_topic = self.get_parameter("run_stage_topic").value
        cancel_topic = self.get_parameter("cancel_topic").value
        reply_topic = self.get_parameter("reply_topic").value
        tts_done_topic = self.get_parameter("tts_done_topic").value
        vlm_api_url = self.get_parameter("vlm_api_url").value
        vlm_timeout = float(self.get_parameter("vlm_timeout").value)
        vlm_temperature = float(self.get_parameter("vlm_temperature").value)
        vlm_max_tokens = int(self.get_parameter("vlm_max_tokens").value)
        history_max = int(self.get_parameter("history_max").value)
        self.use_local_fast_path = bool(
            self.get_parameter("use_local_fast_path").value)
        self.vlm_observe_only = bool(
            self.get_parameter("vlm_observe_only").value)
        self.wait_for_tts_before_action = bool(
            self.get_parameter("wait_for_tts_before_action").value)
        self.tts_wait_timeout = float(
            self.get_parameter("tts_wait_timeout_sec").value)

        self.tts_done_cv = threading.Condition()
        self.tts_done_events = deque(maxlen=100)
        self.cancel_lock = threading.Lock()
        self.cancel_generation = 0
        self.last_cancel_note_time = 0.0
        self.last_director_status = {}

        self.create_subscription(
            String, nl_topic, self._on_nl_command, 10)
        self.create_subscription(
            String, tts_done_topic, self._on_tts_done, 10)
        self.create_subscription(
            String, "/demo/narration_result",
            self._on_narration_result, 10)
        self.create_subscription(
            Empty, "/demo/scene_narrator/clear_history",
            self._on_clear_history, 10)
        self.create_subscription(
            Empty, cancel_topic, self._on_cancel_notice, 10)
        self.create_subscription(
            String, "/demo/director_status",
            self._on_director_status_for_queue, 10)

        self.stage_pub = self.create_publisher(String, stage_topic, 10)
        self.cancel_pub = self.create_publisher(Empty, cancel_topic, 10)
        self.reply_pub = self.create_publisher(String, reply_topic, 10)
        self.mission_end_pub = self.create_publisher(
            Empty, "/demo/mission_end", 10)

        self.memory_lock = threading.Lock()
        self.memory = ConversationMemory(history_max)
        self.task_queue = deque()
        self.queue_lock = threading.Lock()

        self.validator = IntentValidator()
        self.planner = VlmPlanner(
            build_system_prompt(),
            VlmClient(
                vlm_api_url,
                vlm_timeout,
                temperature=vlm_temperature,
                max_tokens=vlm_max_tokens,
            ),
        )
        self.task_executor = TaskExecutor(
            publish_reply=self._publish_reply,
            wait_for_reply=self._publish_reply_and_wait,
            publish_stage=self._publish_stage,
            publish_cancel=self._publish_cancel,
            publish_mission_end=self._publish_mission_end,
            logger=self.get_logger(),
        )

        self.busy_lock = threading.Lock()
        self.busy = False

        self.get_logger().info("════════════════════════════════════════")
        self.get_logger().info("🗣️  NL Parser 节点已启动 (VLM Planner v2)")
        self.get_logger().info(f"  输入话题:   {nl_topic}")
        self.get_logger().info(f"  阶段话题:   {stage_topic}")
        self.get_logger().info(f"  取消话题:   {cancel_topic}")
        self.get_logger().info(f"  回复话题:   {reply_topic}")
        self.get_logger().info(f"  VLM API:    {vlm_api_url}")
        self.get_logger().info(
            f"  本地快速通道: {'启用' if self.use_local_fast_path else '禁用'}")
        self.get_logger().info(
            f"  VLM观察模式: {'启用' if self.vlm_observe_only else '关闭'}")
        self.get_logger().info(
            f"  等待语音:   "
            f"{'启用' if self.wait_for_tts_before_action else '关闭'} "
            f"(done={tts_done_topic}, timeout={self.tts_wait_timeout:.1f}s)")
        self.get_logger().info(
            f"  VLM参数:    temp={vlm_temperature}, max_tokens={vlm_max_tokens}")
        self.get_logger().info(
            "  VLM上下文:  关闭，每次只看当前用户输入")
        self.get_logger().info(
            f"  今日日期:   {today_human()}（已注入 system prompt）")
        self.get_logger().info(
            "  分发策略:   取消即时前置；其余输入走 Planner→Validator→Executor")
        self.get_logger().info("════════════════════════════════════════")

    def _on_nl_command(self, msg: String):
        text = (msg.data or "").strip()
        if not text:
            return
        self.get_logger().info(f"📥 收到: {text!r}")

        if self._is_cancel_text(text):
            self._handle_cancel_text(text)
            return

        if self.use_local_fast_path:
            decision = self.validator.decide_local_fast_path(text)
            if decision is not None:
                threading.Thread(
                    target=self._execute_decision,
                    args=(text, decision),
                    daemon=True,
                ).start()
                return

        with self.busy_lock:
            if self.busy:
                self._publish_reply("⌛ 我还在思考上一条，请稍等再发新指令")
                return
            self.busy = True

        threading.Thread(
            target=self._handle_with_vlm,
            args=(text,),
            daemon=True,
        ).start()

    def _handle_with_vlm(self, text: str):
        try:
            result = self.planner.plan(text)
            parsed = result.parsed
            self.get_logger().info(
                f"🤖 VLM 回复 ({result.latency_sec:.1f}s, 无上下文): "
                f"{result.raw!r}")

            decision = self.validator.decide(text, parsed)
            decision = self._maybe_segmented_decision(text, decision)
            self.get_logger().info(
                f"🧭 Decision: intent={decision.intent}, "
                f"executable={decision.executable}, "
                f"actions={format_stage_actions(decision.actions)}, "
                f"reason={decision.reason}, "
                f"observe_only={self.vlm_observe_only}")

            if self.vlm_observe_only:
                observe_reply = f"🧪 {decision.observe_detail}"
                self._publish_reply(observe_reply)
                self._append_history("user", text)
                self._append_history("assistant", decision.reply or observe_reply)
                return

            self._execute_decision(text, decision)
        except VlmClientError as exc:
            self.get_logger().error(f"VLM 请求失败: {exc}")
            self._publish_reply("⚠️ VLM 不可用，请用面板按钮或标准 stage 命令")
        except Exception as exc:
            self.get_logger().error(f"VLM 解析异常: {exc}")
            self._publish_reply("⚠️ 解析出错，请试试更明确的指令")
        finally:
            with self.busy_lock:
                self.busy = False

    def _maybe_segmented_decision(self, text: str,
                                  whole_decision: Decision) -> Decision:
        segments = segment_tasks(text)
        if len(segments) < 2:
            return whole_decision

        self.get_logger().info(
            f"🧩 尝试多任务分句解析: {len(segments)} 段 {segments}")

        actions = []
        segment_desc = []
        for index, segment in enumerate(segments, 1):
            try:
                result = self.planner.plan(segment)
            except VlmClientError as exc:
                self.get_logger().warn(
                    f"分句 {index}/{len(segments)} VLM 请求失败，"
                    f"回退整句: {segment!r}: {exc}")
                return whole_decision
            except Exception as exc:
                self.get_logger().warn(
                    f"分句 {index}/{len(segments)} 解析异常，回退整句: "
                    f"{segment!r}: {exc}")
                return whole_decision

            decision = self.validator.decide(segment, result.parsed)
            self.get_logger().info(
                f"🧩 分句 {index}/{len(segments)}: {segment!r} -> "
                f"{format_stage_actions(decision.actions)} "
                f"({decision.reason})")
            if not decision.executable or not decision.actions:
                self.get_logger().warn(
                    f"分句 {index}/{len(segments)} 不可执行，回退整句: "
                    f"intent={decision.intent}, reason={decision.reason}")
                return whole_decision

            actions.extend(decision.actions)
            segment_desc.append(
                f"{segment}=>{format_stage_actions(decision.actions)}")

        segmented_score = self._segmented_actions_score(segments, actions)
        whole_score = self._segmented_actions_score(
            segments, whole_decision.actions)
        whole_has_redundancy = self._has_redundant_goto_before_task(
            whole_decision.actions)
        segmented_has_redundancy = self._has_redundant_goto_before_task(
            actions)

        if whole_has_redundancy and not segmented_has_redundancy:
            self.get_logger().info(
                "🧩 分句结果优先：整句解析含冗余 goto，"
                f"segmented={format_stage_actions(actions)}, "
                f"whole={format_stage_actions(whole_decision.actions)}")
        elif segmented_score < whole_score:
            self.get_logger().warn(
                "分句语义完整性低于整句，回退整句: "
                f"segmented={format_stage_actions(actions)}"
                f"(score={segmented_score}), "
                f"whole={format_stage_actions(whole_decision.actions)}"
                f"(score={whole_score})")
            return whole_decision

        repaired_from = (
            f"whole={format_stage_actions(whole_decision.actions)}; "
            f"segments={' | '.join(segment_desc)}")
        return Decision(
            intent="command",
            executable=True,
            actions=actions,
            reply=build_action_reply(actions),
            reason=f"segmented:{len(segments)}",
            repaired_from=repaired_from,
        )

    @staticmethod
    def _has_redundant_goto_before_task(actions: list[StageAction]) -> bool:
        for index in range(len(actions) - 1):
            current = actions[index]
            nxt = actions[index + 1]
            if current.stage == "goto_a" and nxt.waypoint == "A":
                if nxt.stage in {"semantic_find_target", "stage1"}:
                    return True
            if current.stage == "goto_b" and nxt.waypoint == "B":
                if nxt.stage in {"semantic_find_target", "stage2"}:
                    return True
            if current.stage == "goto_c" and nxt.waypoint == "C":
                if nxt.stage in {"semantic_find_target", "stage3"}:
                    return True
        return False

    @staticmethod
    def _segmented_actions_score(segments: list[str],
                                 actions: list[StageAction]) -> int:
        if not segments:
            return 0
        score = 0
        for segment in segments:
            compact = re.sub(r"\s+", "", segment)
            has_find = any(word in compact for word in ("找", "寻找", "寻"))
            has_box = any(word in compact for word in ("箱子", "纸箱", "盒子"))
            has_person = any(
                word in compact
                for word in ("找人", "有人", "人员", "员工", "工作人员", "个人"))
            has_home = any(word in compact for word in ("回家", "归位", "回HOME", "回home"))
            has_goto = any(word in compact for word in ("去", "到", "前往", "开到"))
            waypoint = ""
            if "A" in compact or "a" in compact:
                if "A观测点" in segment or "A点" in segment or "去A" in segment or "到A" in segment:
                    waypoint = "A"
            if "B" in compact or "b" in compact:
                if "B观测点" in segment or "B点" in segment or "去B" in segment or "到B" in segment:
                    waypoint = "B"
            if "C" in compact or "c" in compact:
                if "C观测点" in segment or "C点" in segment or "去C" in segment or "到C" in segment:
                    waypoint = "C"

            matched = False
            for action in actions:
                if has_home and action.stage == "home":
                    matched = True
                    break
                if has_find and has_box:
                    if (action.stage == "semantic_find_target"
                            and action.target_type == "cardboard box"
                            and (not waypoint or action.waypoint == waypoint)):
                        matched = True
                        break
                if has_find and has_person:
                    if action.stage in {"stage1", "stage2", "stage3"}:
                        stage_wp = {"stage1": "A", "stage2": "B", "stage3": "C"}[action.stage]
                        if not waypoint or stage_wp == waypoint:
                            matched = True
                            break
                    if (action.stage == "semantic_find_target"
                            and action.target_type == "person"
                            and (not waypoint or action.waypoint == waypoint)):
                        matched = True
                        break
                if has_goto and not has_find and waypoint:
                    goto_stage = {"A": "goto_a", "B": "goto_b", "C": "goto_c"}[waypoint]
                    if action.stage == goto_stage:
                        matched = True
                        break
            if matched:
                score += 1
        return score

    def _execute_decision(self, user_text: str, decision: Decision):
        if not decision.executable:
            reply = decision.reply or "我还不能确定要执行哪个任务。"
            self._publish_reply(reply)
            self._append_history("user", user_text)
            self._append_history("assistant", reply)
            return

        if not decision.actions:
            reply = decision.reply or "我还不能确定要执行哪个任务。"
            self._publish_reply(reply)
            self._append_history("user", user_text)
            self._append_history("assistant", reply)
            return

        with self.queue_lock:
            self.task_queue.clear()

        first = decision.actions[0]
        rest = decision.actions[1:]
        with self.queue_lock:
            self.task_queue.extend((action, user_text) for action in rest)

        dispatched = self.task_executor.dispatch(
            first, decision.reply, user_request=user_text)
        if dispatched and rest:
            self.get_logger().info(
                f"📋 队列: {format_stage_actions(rest)}")

        self._append_history("user", user_text)
        if decision.reply:
            self._append_history("assistant", decision.reply)

    def _publish_reply(self, text: str):
        msg = String()
        msg.data = text
        self.reply_pub.publish(msg)
        self.get_logger().info(f"💬 {text}")

    def _publish_reply_and_wait(self, text: str) -> bool:
        start_generation = self._get_cancel_generation()
        start_time = time.time()
        self._publish_reply(text)
        return self._wait_for_tts_reply(text, start_generation, start_time)

    def _wait_for_tts_reply(self, text: str, start_generation: int,
                            start_time: float) -> bool:
        if not self.wait_for_tts_before_action:
            return self._get_cancel_generation() == start_generation

        deadline = time.time() + self.tts_wait_timeout
        with self.tts_done_cv:
            while True:
                for event in list(self.tts_done_events):
                    if event.get("source") != "nl_reply":
                        continue
                    raw_text = event.get("raw_text") or event.get("text") or ""
                    if raw_text != text:
                        continue
                    if float(event.get("ts", 0.0) or 0.0) < start_time - 0.1:
                        continue
                    status = event.get("status", "")
                    if status == "interrupted":
                        return False
                    if status != "done":
                        self.get_logger().warn(
                            f"TTS 回复未正常完成: {status} "
                            f"{event.get('detail', '')}")
                    return self._get_cancel_generation() == start_generation

                if self._get_cancel_generation() != start_generation:
                    return False
                remaining = deadline - time.time()
                if remaining <= 0:
                    self.get_logger().warn(
                        f"等待 TTS 回复完成超时: {text[:40]}")
                    return self._get_cancel_generation() == start_generation
                self.tts_done_cv.wait(timeout=min(0.2, remaining))

    def _publish_stage(self, stage: str, target_object=None,
                       user_request: str = "", waypoint: str = "",
                       target_description: str = "",
                       target_type: str = ""):
        msg = String()
        if (target_object or user_request or waypoint or target_description
                or target_type):
            payload = {"stage": stage}
            if target_object:
                payload["target_object"] = target_object
            if user_request:
                payload["user_request"] = user_request
            if waypoint:
                payload["waypoint"] = waypoint
            if target_description:
                payload["target_description"] = target_description
            if target_type:
                payload["target_type"] = target_type
            msg.data = json.dumps(payload, ensure_ascii=False)
        else:
            msg.data = stage
        self.stage_pub.publish(msg)
        self.get_logger().info(
            f"▶ 转发到 /demo/run_stage: {stage}"
            f"{f' [{target_object}]' if target_object else ''}"
            f"{f' @{waypoint}' if waypoint else ''}"
            f"{f' <{target_type}>' if target_type else ''}"
            f"{f' ({target_description})' if target_description else ''}")

    def _publish_cancel(self):
        self.cancel_pub.publish(Empty())

    def _publish_mission_end(self):
        self.mission_end_pub.publish(Empty())

    def _on_tts_done(self, msg: String):
        try:
            event = json.loads(msg.data) if msg.data else {}
        except json.JSONDecodeError:
            return
        with self.tts_done_cv:
            self.tts_done_events.append(event)
            self.tts_done_cv.notify_all()

    def _on_cancel_notice(self, _msg):
        with self.queue_lock:
            self.task_queue.clear()
        with self.cancel_lock:
            self.cancel_generation += 1
            now = time.time()
            should_log = now - self.last_cancel_note_time >= 0.5
            self.last_cancel_note_time = now
        with self.tts_done_cv:
            self.tts_done_cv.notify_all()
        if should_log:
            with self.memory_lock:
                self.memory.append_cancel_observation(
                    self.last_director_status or {})
                count = len(self.memory)
            self.get_logger().info(f"📚 历史[{count}] +cancel_observation")

    def _on_narration_result(self, msg: String):
        with self.memory_lock:
            item = self.memory.append_narration_result(msg.data)
            count = len(self.memory)
        if item:
            self.get_logger().info(
                f"📚 历史[{count}] +observation: {item[:80]}")

    def _on_clear_history(self, _msg):
        with self.memory_lock:
            self.memory.clear()
        self.get_logger().info("🧹 对话历史已清空")

    def _on_director_status_for_queue(self, msg: String):
        try:
            obj = json.loads(msg.data) if msg.data else {}
        except json.JSONDecodeError:
            return
        self.last_director_status = obj
        step = str(obj.get("step", "")).strip()
        if step != "end":
            return

        message = str(obj.get("message", "")).strip()
        if "中断" in message or "失败" in message:
            with self.queue_lock:
                self.task_queue.clear()
            self._append_history(
                "assistant",
                f"(系统观察) 任务未完成：{message}；后续队列已取消。")
            return

        with self.queue_lock:
            if not self.task_queue:
                return
            queued = self.task_queue.popleft()
            remaining = len(self.task_queue)
        if isinstance(queued, tuple):
            next_action, user_request = queued
        else:
            next_action = queued
            user_request = ""

        self.get_logger().info(
            f"📋 队列自动发下一个: {next_action.stage}"
            f"{f' [{next_action.target_object}]' if next_action.target_object else ''}"
            f"（剩余 {remaining} 项）")
        reply = "上一步完成，继续执行：" + (
            build_action_reply([next_action]).removeprefix("好的，")
        )
        threading.Thread(
            target=self.task_executor.dispatch,
            args=(next_action, reply, user_request),
            daemon=True,
        ).start()

    def _handle_cancel_text(self, text: str):
        self._publish_cancel()
        reply = "🛑 已取消当前任务"
        self._publish_reply(reply)
        self._append_history("user", text)
        self._append_history("assistant", reply)

    def _append_history(self, role: str, content: str):
        with self.memory_lock:
            self.memory.append(role, content)
            count = len(self.memory)
        self.get_logger().info(
            f"📚 历史[{count}] +{role}: {content[:60]}")

    def _get_cancel_generation(self) -> int:
        with self.cancel_lock:
            return self.cancel_generation

    @staticmethod
    def _is_cancel_text(text: str) -> bool:
        return any(re.search(pat, text, re.IGNORECASE)
                   for pat in CANCEL_PATTERNS)


def main(args=None):
    rclpy.init(args=args)
    node = NlParserNode()
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
