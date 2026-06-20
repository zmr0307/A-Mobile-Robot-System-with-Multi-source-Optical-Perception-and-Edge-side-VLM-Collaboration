import json
from collections import deque


class ConversationMemory:
    def __init__(self, maxlen: int):
        self._items = deque(maxlen=int(maxlen))

    def append(self, role: str, content: str):
        self._items.append({"role": role, "content": content})

    def clear(self):
        self._items.clear()

    def __len__(self):
        return len(self._items)

    def append_cancel_observation(self, status: dict):
        stage = str(status.get("stage", "") or "").strip()
        step = str(status.get("step", "") or "").strip()
        detail = "用户取消/中断了当前任务"
        if stage or step:
            detail += f"（取消时状态: {stage or '-'} / {step or '-'}）"
        detail += "；取消后的动作未继续执行，不能视为已到达或已完成。"
        self.append("assistant", f"(系统观察) {detail}")

    def append_narration_result(self, raw: str):
        try:
            obj = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            return None

        trigger = obj.get("trigger", {}) or {}
        mode = str(trigger.get("mode", "")).strip() or "unknown"
        wp = str(trigger.get("waypoint", "")).strip() or "当前位置"
        has_person = bool(obj.get("has_person", False))
        desc = str(obj.get("description", "")).strip()
        key_objects = obj.get("key_objects", []) or []

        person_tag = "发现人员" if has_person else "未发现人员"
        head = f"(系统观察) {mode}@{wp}: {person_tag}"
        if isinstance(key_objects, list) and key_objects:
            head += f"，画面有 {'、'.join(str(x) for x in key_objects[:5])}"
        if desc:
            head += f"。{desc[:120]}"
        self.append("assistant", head)
        return head
