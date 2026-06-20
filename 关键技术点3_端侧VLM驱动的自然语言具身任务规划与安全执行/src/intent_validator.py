import re

from wl100_demo.nlu_schema import Decision, ParsedIntent, StageAction
from wl100_demo.skill_mapper import (
    actions_from_legacy,
    build_action_reply,
    format_stage_actions,
    map_steps_to_actions,
)


_COMMAND_LIKE_RE = re.compile(
    r"去|到|前往|开到|带我|找人|寻人|有没有人|有人吗|"
    r"人在哪|人在哪里|人在不在|寻个人|找一个人|"
    r"工作人员|员工|目标人员|目标人|找箱子|找纸箱|找物品|"
    r"送|交接|拿|回家|归位|巡检|看看|看一下|观察|查看|"
    r"检测|扫一眼|描述|生成报告|出报告|取消|停止|别走了|"
    r"回HOME|回home|回初始点|初始点|回原点|回起点|跑一遍|都跑"
)
_CAPABILITY_QUERY_RE = re.compile(
    r"能不能|可以吗|能否|会不会|能去吗|能到吗|能找吗|"
    r"你能|你会|你可以|是否可以|可以.{0,12}吗"
)
_MISSING_FIND_TARGET_RE = re.compile(
    r"(去|到|前往|开到|帮我|带我)?.{0,4}(找人|寻人|找一下人|找个人|工作人员)"
)
_MISSING_SEMANTIC_TARGET_RE = re.compile(
    r"(找|寻找|寻|看看有没有|看有没有|有没有).{0,40}"
    r"(人|人员|工作人员|员工|纸箱|箱子|盒子|椅子|桌子|桌|门|包|背包|手机)"
)
_MISSING_GOTO_TARGET_RE = re.compile(r"^(去|到|前往|开到|带我去)$")
_GO_LOOK_WITHOUT_WAYPOINT_RE = re.compile(
    r"^(去|到|前往|开到).{0,4}(看看|看一下|看下|观察|查看|瞧瞧)$"
)
_USER_SEMANTIC_FIND_RE = re.compile(
    r"(找|寻|看看有没有|有没有).{0,16}"
    r"(穿|戴|拿|握|抱|背|坐|站|蹲|躺|靠|离|附近|旁边|最近|"
    r"左边|右边|前面|后面|黑|白|红|蓝|绿|黄|帽|裤|衣|鞋|包|"
    r"箱|手机|椅|桌|门|工作服|工牌).{0,16}人|"
    r"(穿|戴|拿|握|抱|背|坐|站|蹲|躺|靠|离|附近|旁边|最近|"
    r"左边|右边|前面|后面|黑|白|红|蓝|绿|黄|帽|裤|衣|鞋|包|"
    r"箱|手机|椅|桌|门|工作服|工牌).{0,16}人"
)
_HANDOVER_TEXT_RE = re.compile(r"送|交接|递给|交给|放到|拿给")
_WAYPOINT_LETTER_RE = re.compile(r"(?<![a-zA-Z])([abcABC])(?![a-zA-Z])")
_SEMANTIC_FIND_DESC_RE = re.compile(
    r"(?:找一下|找|寻找|寻|看看有没有|看有没有|确认有没有|有没有)"
    r"(?:一个|一位|个|位|那个|这个|这位|目标)?"
    r"(.{1,48}?人)"
)
_SEMANTIC_FIND_DESC_FALLBACK_RE = re.compile(
    r"((?:穿|戴|拿|握|抱|背|坐|站|蹲|躺|靠|离|在|附近|旁边|最近|"
    r"左边|右边|前面|后面|黑|白|红|蓝|绿|黄|帽|裤|衣|鞋|包|"
    r"箱|纸箱|手机|椅|桌|门|工作服|工牌).{0,48}?人)"
)
_SEMANTIC_TARGET_DESC_RE = re.compile(
    r"(?:找一下|找|寻找|寻|看看有没有|看有没有|确认有没有|有没有)"
    r"(?:一个|一位|个|位|那个|这个|这位|目标)?"
    r"(.{0,64})"
)
_PERSON_TARGET_PATTERN = r"工作人员|员工|人员|目标人员|目标人|(?<!机器)(?<!无)人"
_SEMANTIC_FIND_REPAIRABLE_STAGES = {
    "stage1", "stage2", "stage3",
    "goto_a", "goto_b", "goto_c",
    "inspect_here",
}


class IntentValidator:
    """统一把 VLM 语义结果转成可执行 Decision。

    VLM 是主理解器；本地 Validator 负责缺槽位追问、合法性检查和
    可执行动作映射，不能绕过 Decision 直接下发任务。
    """

    def __init__(self):
        pass

    def decide(self, text: str, parsed: ParsedIntent) -> Decision:
        if _CAPABILITY_QUERY_RE.search(text):
            return Decision(
                intent="chat",
                executable=False,
                reply=self._capability_reply(text),
                reason="capability_query",
            )

        local_actions = self._local_fast_path_actions(text)
        semantic_repair_actions = self._semantic_find_repair_actions(text)
        mapped_actions, issues = map_steps_to_actions(parsed.steps)
        mapped_actions = self._drop_redundant_inspect(mapped_actions)

        if parsed.intent == "command":
            if mapped_actions:
                if self._inspect_needs_waypoint(text, mapped_actions):
                    return Decision(
                        intent="clarify",
                        executable=False,
                        reply="你想让我去哪个观测点看看？A、B、C 里选一个。",
                        reason="missing_goto_waypoint",
                    )
                if (semantic_repair_actions
                        and self._should_repair_to_semantic_find(
                            mapped_actions)):
                    return Decision(
                        intent="command",
                        executable=True,
                        actions=semantic_repair_actions,
                        reply=build_action_reply(semantic_repair_actions),
                        reason="semantic_find_text_repair",
                        repaired_from=format_stage_actions(mapped_actions),
                    )
                if (semantic_repair_actions
                        and self._semantic_repair_corrects_target_type(
                            mapped_actions, semantic_repair_actions)):
                    return Decision(
                        intent="command",
                        executable=True,
                        actions=semantic_repair_actions,
                        reply=build_action_reply(semantic_repair_actions),
                        reason="semantic_find_target_type_repair",
                        repaired_from=format_stage_actions(mapped_actions),
                    )
                protect_semantic = self._protect_semantic_find(
                    text, mapped_actions)
                if (self._has_semantic_find_person(mapped_actions)
                        and not protect_semantic and local_actions):
                    return Decision(
                        intent="command",
                        executable=True,
                        actions=local_actions,
                        reply=build_action_reply(det_actions),
                        reason="local_semantic_guard",
                        repaired_from=format_stage_actions(mapped_actions),
                    )
                if self._local_waypoint_over_inspect(
                        mapped_actions, local_actions):
                    return Decision(
                        intent="command",
                        executable=True,
                        actions=local_actions,
                        reply=build_action_reply(det_actions),
                        reason="local_waypoint_refine",
                        repaired_from=format_stage_actions(mapped_actions),
                    )
                if (not protect_semantic
                        and self._local_action_is_more_specific(
                            mapped_actions, local_actions)):
                    return Decision(
                        intent="command",
                        executable=True,
                        actions=local_actions,
                        reply=build_action_reply(det_actions),
                        reason="local_refine",
                        repaired_from=format_stage_actions(mapped_actions),
                    )
                return Decision(
                    intent="command",
                    executable=True,
                    actions=mapped_actions,
                    reply=build_action_reply(mapped_actions),
                    reason="vlm_steps",
                )
            if semantic_repair_actions:
                return Decision(
                    intent="command",
                    executable=True,
                    actions=semantic_repair_actions,
                    reply=build_action_reply(semantic_repair_actions),
                    reason="semantic_find_text_repair",
                    repaired_from=_steps_desc(parsed),
                )
            if local_actions:
                return Decision(
                    intent="command",
                    executable=True,
                    actions=local_actions,
                    reply=build_action_reply(det_actions),
                    reason="local_repair",
                    repaired_from=_steps_desc(parsed),
                )
            return self._clarify_for_command(text, parsed.reply, issues)

        if (semantic_repair_actions
                and self._can_repair_non_command(text, parsed)):
            return Decision(
                intent="command",
                executable=True,
                actions=semantic_repair_actions,
                reply=build_action_reply(semantic_repair_actions),
                reason=f"semantic_find_non_command_repair:{parsed.intent}",
                repaired_from=parsed.reply or parsed.intent,
            )

        if local_actions and self._can_repair_non_command(text, parsed):
            return Decision(
                intent="command",
                executable=True,
                actions=local_actions,
                reply=build_action_reply(det_actions),
                reason=f"local_non_command_repair:{parsed.intent}",
                repaired_from=parsed.reply or parsed.intent,
            )

        if self._should_clarify_non_command(text, parsed):
            return self._clarify_for_command(text, parsed.reply, [])

        if parsed.intent == "clarify":
            return Decision(
                intent="clarify",
                executable=False,
                reply=parsed.reply or self._fallback_clarify(text),
                reason="vlm_clarify",
            )

        return Decision(
            intent=parsed.intent if parsed.intent else "chat",
            executable=False,
            reply=parsed.reply or "嗯，我听到了。",
            reason="non_command",
        )

    def decide_local_fast_path(self, text: str) -> Decision | None:
        actions = self._local_fast_path_actions(text)
        if not actions:
            return None
        return Decision(
            intent="command",
            executable=True,
            actions=actions,
            reply=build_action_reply(actions),
            reason="local_fast_path",
        )

    def _local_fast_path_actions(self, text: str):
        """保留本地快速出口，但公开展示版不依赖额外规则解析器。"""
        return []

    @staticmethod
    def _semantic_find_repair_actions(text: str):
        if _HANDOVER_TEXT_RE.search(text):
            return []
        if not (_USER_SEMANTIC_FIND_RE.search(text)
                or _MISSING_SEMANTIC_TARGET_RE.search(text)):
            return []
        waypoint = _extract_waypoint_letter(text)
        if waypoint is None:
            return []
        desc = _extract_semantic_target_description(text)
        if not desc:
            return []
        target_type = _infer_target_type(desc)
        if not target_type:
            return []
        if target_type == "person" and _is_generic_person_text(desc):
            find_stage = {"A": "stage1", "B": "stage2", "C": "stage3"}[waypoint]
            return [StageAction(find_stage)]
        return [StageAction(
            "semantic_find_target",
            waypoint=waypoint,
            target_description=desc,
            target_type=target_type,
        )]

    def _can_repair_non_command(self, text: str, parsed: ParsedIntent) -> bool:
        if _CAPABILITY_QUERY_RE.search(text):
            return False
        if parsed.intent in ("query_history", "chat") and "?" in text:
            return False
        if parsed.reply and parsed.reply.startswith(("好的", "收到", "可以")):
            return True
        return bool(_COMMAND_LIKE_RE.search(text))

    @staticmethod
    def _has_semantic_find_person(actions) -> bool:
        return any(
            item.stage in {"semantic_find_person", "semantic_find_target"}
            and item.target_description
            and (not item.target_type or item.target_type == "person")
            for item in actions
        )

    @staticmethod
    def _protect_semantic_find(text: str, actions) -> bool:
        return (
            IntentValidator._has_semantic_find_person(actions)
            and _USER_SEMANTIC_FIND_RE.search(text)
        )

    @staticmethod
    def _local_waypoint_over_inspect(mapped_actions,
                                            det_actions) -> bool:
        return (
            len(mapped_actions) == 1
            and mapped_actions[0].stage == "inspect_here"
            and any(_action_waypoint(action) in {"A", "B", "C"}
                    for action in det_actions)
        )

    @staticmethod
    def _inspect_needs_waypoint(text: str, actions) -> bool:
        if len(actions) != 1 or actions[0].stage != "inspect_here":
            return False
        if re.search(r"(?<![a-zA-Z])[abcABC](?![a-zA-Z])", text):
            return False
        if re.search(r"原地|这里|这边|当前|周围|附近|现场|四周", text):
            return False
        return bool(_GO_LOOK_WITHOUT_WAYPOINT_RE.search(text.strip()))

    @staticmethod
    def _local_action_is_more_specific(mapped_actions, det_actions) -> bool:
        if not det_actions:
            return False

        mapped_targets = sum(1 for item in mapped_actions if item.target_object)
        det_targets = sum(1 for item in det_actions if item.target_object)
        if det_targets > mapped_targets:
            return True

        for mapped, det in zip(mapped_actions, det_actions):
            if mapped.stage == det.stage:
                continue
            mapped_wp = _action_waypoint(mapped)
            det_wp = _action_waypoint(det)
            if mapped_wp and mapped_wp == det_wp:
                mapped_rank = _action_specificity(mapped)
                det_rank = _action_specificity(det)
                if det_rank > mapped_rank:
                    return True
                if det_rank < mapped_rank:
                    return False
        return len(det_actions) > len(mapped_actions)

    @staticmethod
    def _should_repair_to_semantic_find(mapped_actions) -> bool:
        if len(mapped_actions) != 1:
            return False
        action = mapped_actions[0]
        if action.stage in {"semantic_find_person", "semantic_find_target"}:
            return False
        return action.stage in _SEMANTIC_FIND_REPAIRABLE_STAGES

    @staticmethod
    def _semantic_repair_corrects_target_type(mapped_actions,
                                              repair_actions) -> bool:
        if len(mapped_actions) != 1 or len(repair_actions) != 1:
            return False
        mapped = mapped_actions[0]
        repair = repair_actions[0]
        if repair.stage != "semantic_find_target" or not repair.target_type:
            return False
        if mapped.stage not in {"semantic_find_person", "semantic_find_target"}:
            return False
        if _action_waypoint(mapped) != _action_waypoint(repair):
            return False
        mapped_type = mapped.target_type or _infer_target_type(
            mapped.target_description)
        repair_type = repair.target_type or _infer_target_type(
            repair.target_description)
        return bool(mapped_type and repair_type and mapped_type != repair_type)

    @staticmethod
    def _drop_redundant_inspect(actions):
        if len(actions) <= 1:
            return actions
        has_waypoint_task = any(
            item.stage in {
                "stage1", "stage2", "stage3",
                "goto_a", "goto_b", "goto_c",
            }
            for item in actions
        )
        if not has_waypoint_task:
            return actions
        return [item for item in actions if item.stage != "inspect_here"]

    def _should_clarify_non_command(self, text: str,
                                    parsed: ParsedIntent) -> bool:
        if parsed.intent == "query_history":
            return False
        stripped = text.strip()
        command_prefix = stripped.startswith(
            ("去", "到", "前往", "开到", "带我", "帮我", "找", "寻"))
        misleading_ack = bool(parsed.reply) and parsed.reply.startswith(
            ("好的", "收到", "可以"))
        return (
            (self._looks_like_find_without_waypoint(text, [])
             or self._looks_like_semantic_target_without_waypoint(text, []))
            and (command_prefix or misleading_ack)
        )

    def _clarify_for_command(self, text: str, reply: str, issues: list[str]):
        if self._looks_like_semantic_target_without_waypoint(text, issues):
            return Decision(
                intent="clarify",
                executable=False,
                reply="你想让我去哪个观测点找这个目标？A、B、C 里选一个。",
                reason="missing_semantic_find_target_waypoint",
            )
        if self._looks_like_find_without_waypoint(text, issues):
            return Decision(
                intent="clarify",
                executable=False,
                reply="你想让我去哪个观测点找人？A、B、C 里选一个。",
                reason="missing_find_waypoint",
            )
        if "semantic_find_person_missing_waypoint" in issues:
            return Decision(
                intent="clarify",
                executable=False,
                reply="你想让我去哪个观测点找这个人？A、B、C 里选一个。",
                reason="missing_semantic_find_person_waypoint",
            )
        if "semantic_find_person_missing_description" in issues:
            return Decision(
                intent="clarify",
                executable=False,
                reply="你想找什么样的人？请描述衣着、姿态、动作或位置关系。",
                reason="missing_semantic_find_person_description",
            )
        if "semantic_find_target_missing_waypoint" in issues:
            return Decision(
                intent="clarify",
                executable=False,
                reply="你想让我去哪个观测点找这个目标？A、B、C 里选一个。",
                reason="missing_semantic_find_target_waypoint",
            )
        if "semantic_find_target_missing_type" in issues:
            return Decision(
                intent="clarify",
                executable=False,
                reply="你想找什么目标？请说清楚是人、箱子还是其它物品。",
                reason="missing_semantic_find_target_type",
            )
        if self._looks_like_goto_without_waypoint(text, issues):
            return Decision(
                intent="clarify",
                executable=False,
                reply="你想让我去哪个观测点？A、B、C 里选一个。",
                reason="missing_goto_waypoint",
            )
        if "handover_missing_object" in issues:
            return Decision(
                intent="clarify",
                executable=False,
                reply="目前交接只支持箱子。你是要我去哪个观测点送箱子？",
                reason="missing_handover_object",
            )
        if "handover_missing_waypoint" in issues:
            return Decision(
                intent="clarify",
                executable=False,
                reply="你想让我去哪个观测点交接箱子？A、B、C 里选一个。",
                reason="missing_handover_waypoint",
            )
        return Decision(
            intent="clarify",
            executable=False,
            reply=reply or self._fallback_clarify(text),
            reason="command_without_safe_action",
        )

    @staticmethod
    def _looks_like_find_without_waypoint(text: str, issues: list[str]) -> bool:
        return (
            "find_person_missing_waypoint" in issues
            or (_MISSING_FIND_TARGET_RE.search(text)
                and not re.search(r"(?<![a-zA-Z])[abcABC](?![a-zA-Z])", text))
        )

    @staticmethod
    def _looks_like_semantic_target_without_waypoint(
            text: str, issues: list[str]) -> bool:
        object_target = re.search(
            r"纸箱|箱子|盒子|椅子|桌子|桌|门|背包|包|手机",
            text,
        )
        return (
            "semantic_find_target_missing_waypoint" in issues
            or (object_target
                and _MISSING_SEMANTIC_TARGET_RE.search(text)
                and not re.search(r"(?<![a-zA-Z])[abcABC](?![a-zA-Z])", text))
        )

    @staticmethod
    def _looks_like_goto_without_waypoint(text: str, issues: list[str]) -> bool:
        return "goto_missing_waypoint" in issues or bool(
            _MISSING_GOTO_TARGET_RE.search(text.strip())
        )

    @staticmethod
    def _fallback_clarify(text: str) -> str:
        if _MISSING_FIND_TARGET_RE.search(text):
            return "你想让我去哪个观测点找人？A、B、C 里选一个。"
        return "我还不能确定要执行哪个任务，请说清楚观测点和动作。"

    @staticmethod
    def _capability_reply(text: str) -> str:
        match = re.search(r"(?<![a-zA-Z])([abcABC])(?![a-zA-Z])", text)
        if match:
            wp = match.group(1).upper()
            return (
                f"可以，我能去{wp}观测点。"
                f"如果要我现在过去，请直接说：去{wp}观测点。"
            )
        return "可以。你如果要我现在执行任务，请直接说观测点和动作。"


def _steps_desc(parsed: ParsedIntent) -> str:
    if not parsed.steps:
        return "[]"
    return "[" + ", ".join(
        f"{step.kind}@{step.waypoint or '-'}"
        f"{':' + step.object if step.object else ''}"
        f"{'<' + step.target_type + '>' if step.target_type else ''}"
        for step in parsed.steps
    ) + "]"


def _extract_waypoint_letter(text: str) -> str | None:
    match = _WAYPOINT_LETTER_RE.search(text)
    return match.group(1).upper() if match else None


def _extract_semantic_person_description(text: str) -> str:
    match = _SEMANTIC_FIND_DESC_RE.search(text)
    if not match:
        match = _SEMANTIC_FIND_DESC_FALLBACK_RE.search(text)
    if not match:
        return ""
    desc = match.group(1)
    desc = desc.strip(" ，,。.!！？?；;：:")
    desc = re.sub(r"^(一个|一位|个|位|那个|这个|这位|目标)", "", desc)
    return desc.strip(" ，,。.!！？?；;：:")


def _extract_semantic_target_description(text: str) -> str:
    match = _SEMANTIC_TARGET_DESC_RE.search(text)
    if match:
        desc = match.group(1)
    else:
        desc = _extract_semantic_person_description(text)
    desc = re.split(
        r"然后|接着|再去|再|最后|之后|并且|同时|回家|回HOME|回home",
        desc or "",
        maxsplit=1,
    )[0]
    desc = desc.strip(" ，,。.!！？?；;：:")
    desc = re.sub(r"^(一个|一位|个|位|那个|这个|这位|目标)", "", desc)
    return desc.strip(" ，,。.!！？?；;：:")


def _infer_target_type(desc: str) -> str | None:
    text = str(desc or "")
    patterns = [
        ("person", _PERSON_TARGET_PATTERN),
        ("cardboard box", r"纸箱|箱子|盒子|box|carton"),
        ("chair", r"椅子|chair"),
        ("table", r"桌子|桌|table|desk"),
        ("bag", r"背包|包|bag|backpack"),
        ("phone", r"手机|phone"),
        ("door", r"门|door"),
    ]
    matches = []
    for target_type, pattern in patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            matches.append((match.start(), match.end(), target_type))
    if matches:
        matches.sort(key=lambda item: (item[1], item[0]))
        return matches[-1][2]
    return None


def _is_generic_person_text(desc: str) -> bool:
    compact = re.sub(r"\s|，|,|。", "", desc or "")
    return compact in {
        "人", "人员", "工作人员", "员工", "一个人", "一个人员",
        "目标人", "目标人员",
    }


def _action_waypoint(action) -> str | None:
    find_wp = {"stage1": "A", "stage2": "B", "stage3": "C"}
    goto_wp = {"goto_a": "A", "goto_b": "B", "goto_c": "C"}
    return action.waypoint or find_wp.get(action.stage) or goto_wp.get(action.stage)


def _action_specificity(action) -> int:
    if (action.stage in {"semantic_find_person", "semantic_find_target"}
            and action.target_description):
        return 3
    if action.stage in {"stage1", "stage2", "stage3"}:
        return 3 if action.target_object else 2
    if action.stage in {"goto_a", "goto_b", "goto_c"}:
        return 1
    return 2
