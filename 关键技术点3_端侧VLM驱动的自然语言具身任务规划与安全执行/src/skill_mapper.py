import re

from wl100_demo.nlu_schema import (
    HANDOVER_OBJECTS,
    STAGE_NAMES,
    StageAction,
    SemanticStep,
    format_stage_actions,
)


_WAYPOINT_TO_FIND = {"A": "stage1", "B": "stage2", "C": "stage3"}
_WAYPOINT_TO_GOTO = {"A": "goto_a", "B": "goto_b", "C": "goto_c"}
_FIND_TO_WAYPOINT = {"stage1": "A", "stage2": "B", "stage3": "C"}
_GOTO_TO_WAYPOINT = {"goto_a": "A", "goto_b": "B", "goto_c": "C"}
_FEATURE_WORD_RE = re.compile(
    r"穿|戴|拿|握|抱|背|坐|站|蹲|躺|靠|旁边|附近|最近|左边|右边|"
    r"前面|后面|黑|白|红|蓝|绿|黄|帽|裤|衣|鞋|包|箱|手机|椅|桌|门|"
    r"工作服|工牌"
)
_GENERIC_PERSON_RE = re.compile(
    r"^(找|寻|找一下|寻找|看有没有|看看有没有|有没有)?"
    r"(一个|一位|个|位|某个|某位|具体)?"
    r"(人|个人|人员|工作人员|员工|目标人|目标人员)$"
)
_PERSON_TARGET_PATTERN = r"工作人员|员工|人员|目标人员|目标人|(?<!机器)(?<!无)人"


def map_steps_to_actions(steps: list[SemanticStep]):
    """把 VLM 语义 steps 转为 director 可执行 stage。

    返回 (actions, issues)。issues 是给 validator 用的缺槽位/非法槽位说明。
    """
    actions: list[StageAction] = []
    issues: list[str] = []

    for step in steps:
        kind = (step.kind or "").strip().lower()
        waypoint = (step.waypoint or "").strip().upper() or None
        obj = (step.object or "").strip() or None
        target_description = (
            (step.target_description or "").strip() or None
        )

        if kind == "goto":
            if waypoint not in _WAYPOINT_TO_GOTO:
                issues.append("goto_missing_waypoint")
                continue
            actions.append(StageAction(_WAYPOINT_TO_GOTO[waypoint]))
        elif kind == "find_person":
            if waypoint not in _WAYPOINT_TO_FIND:
                issues.append("find_person_missing_waypoint")
                continue
            actions.append(StageAction(_WAYPOINT_TO_FIND[waypoint]))
        elif kind == "semantic_find_person":
            if waypoint not in _WAYPOINT_TO_FIND:
                issues.append("semantic_find_person_missing_waypoint")
                continue
            if _is_generic_person_description(target_description):
                actions.append(StageAction(_WAYPOINT_TO_FIND[waypoint]))
                continue
            if not target_description:
                issues.append("semantic_find_person_missing_description")
                continue
            actions.append(StageAction(
                "semantic_find_target",
                waypoint=waypoint,
                target_description=target_description,
                target_type="person",
            ))
        elif kind == "semantic_find_target":
            if waypoint not in _WAYPOINT_TO_FIND:
                issues.append("semantic_find_target_missing_waypoint")
                continue
            target_type = _normalise_target_type(
                step.target_type or target_description)
            if not target_type:
                issues.append("semantic_find_target_missing_type")
                continue
            if target_type == "person" and _is_generic_person_description(
                    target_description):
                actions.append(StageAction(_WAYPOINT_TO_FIND[waypoint]))
                continue
            desc = target_description or _target_type_label(target_type)
            actions.append(StageAction(
                "semantic_find_target",
                waypoint=waypoint,
                target_description=desc,
                target_type=target_type,
            ))
        elif kind == "handover":
            if waypoint not in _WAYPOINT_TO_FIND:
                issues.append("handover_missing_waypoint")
                continue
            if obj not in HANDOVER_OBJECTS:
                issues.append("handover_missing_object")
                continue
            actions.append(StageAction(_WAYPOINT_TO_FIND[waypoint], obj))
        elif kind == "inspect_here":
            if waypoint:
                issues.append("inspect_here_with_waypoint")
                continue
            actions.append(StageAction("inspect_here"))
        elif kind == "home":
            actions.append(StageAction("home"))
        elif kind == "patrol_all":
            actions.append(StageAction("patrol_all"))
        elif kind == "generate_report":
            actions.append(StageAction("generate_report"))
        elif kind == "cancel":
            actions.append(StageAction("cancel"))
        else:
            issues.append(f"unknown_kind:{kind}")

    return dedupe_actions(actions), issues


def actions_from_legacy(raw_actions) -> list[StageAction]:
    actions: list[StageAction] = []
    for item in raw_actions or []:
        if not isinstance(item, dict):
            continue
        stage = str(item.get("stage", "")).strip()
        target = item.get("target_object")
        target = str(target).strip() if target else None
        waypoint = item.get("waypoint")
        waypoint = str(waypoint).strip().upper() if waypoint else None
        target_description = item.get("target_description")
        target_description = (
            str(target_description).strip() if target_description else None
        )
        target_type = item.get("target_type")
        target_type = _normalise_target_type(target_type or target_description)
        if stage in STAGE_NAMES:
            actions.append(StageAction(
                stage,
                target,
                waypoint=waypoint,
                target_description=target_description,
                target_type=target_type,
            ))
    return dedupe_actions(actions)


def actions_to_legacy_dicts(actions: list[StageAction]):
    return [action.as_dict() for action in actions]


def dedupe_actions(actions: list[StageAction]) -> list[StageAction]:
    deduped: list[StageAction] = []
    for action in actions:
        if not deduped:
            deduped.append(action)
            continue
        prev = deduped[-1]
        if prev.stage == action.stage:
            if action.stage == "semantic_find_target":
                same_target = (
                    prev.waypoint == action.waypoint
                    and prev.target_type == action.target_type
                    and prev.target_description == action.target_description
                )
                if same_target:
                    continue
                deduped.append(action)
                continue
            if action.target_object and not prev.target_object:
                deduped[-1] = action
            continue
        if action.stage in _GOTO_TO_WAYPOINT:
            wp = _GOTO_TO_WAYPOINT[action.stage]
            if prev.stage == _WAYPOINT_TO_FIND.get(wp):
                continue
        if action.stage in _FIND_TO_WAYPOINT:
            wp = _FIND_TO_WAYPOINT[action.stage]
            if prev.stage == _WAYPOINT_TO_GOTO.get(wp):
                deduped[-1] = action
                continue
        if action.stage == "semantic_find_target" and action.waypoint:
            if prev.stage == _WAYPOINT_TO_GOTO.get(action.waypoint):
                deduped[-1] = action
                continue
        deduped.append(action)
    return deduped


def build_action_reply(actions: list[StageAction]) -> str:
    labels = {
        "stage1": "A观测点找人",
        "stage2": "B观测点找人",
        "stage3": "C观测点找人",
        "semantic_find_person": "语义找人",
        "semantic_find_target": "语义找目标",
        "goto_a": "去A观测点",
        "goto_b": "去B观测点",
        "goto_c": "去C观测点",
        "home": "归位",
        "patrol_all": "巡检全部观测点",
        "inspect_here": "查看周围",
        "generate_report": "生成报告",
        "cancel": "取消当前任务",
    }
    parts = []
    for action in actions:
        if action.stage in {"semantic_find_person", "semantic_find_target"}:
            wp = action.waypoint or "指定"
            desc = (
                action.target_description
                or _target_type_label(action.target_type)
                or "目标"
            )
            text = f"去{wp}观测点找{desc}"
            parts.append(text)
            continue
        text = labels.get(action.stage, action.stage)
        if action.target_object:
            text += f"交接{action.target_object}"
        parts.append(text)
    return "好的，" + "，".join(parts) + "。"


def _is_generic_person_description(text: str | None) -> bool:
    """没有外观/姿态/动作等约束时，保持旧的普通找人流程。"""
    if not text:
        return True
    compact = (
        text.replace(" ", "")
            .replace("，", "")
            .replace(",", "")
            .replace("。", "")
    )
    compact = re.sub(
        r"^(帮我|麻烦|请)?(去|到|前往)?[abcABC]"
        r"(观测点|点|那边|那里|这边)?",
        "",
        compact,
    )
    compact = re.sub(r"^(帮我|麻烦|请|去|到|前往)", "", compact)
    compact = compact.replace("寻找", "找")
    if _FEATURE_WORD_RE.search(compact):
        return False
    if ("自然语言描述" in compact or "符合描述" in compact) and "人" in compact:
        return True
    if "具体" in compact and ("人" in compact or "人员" in compact):
        return True
    return compact in {
        "人",
        "找人",
        "找个人",
        "找一个人",
        "找一个个人",
        "一个人",
        "一个个人",
        "某个人",
        "目标人",
        "目标人员",
        "工作人员",
        "找工作人员",
        "员工",
        "找员工",
        "具体的人",
        "一个具体的人",
        "找一个具体的人",
    } or bool(_GENERIC_PERSON_RE.fullmatch(compact))


def _normalise_target_type(value: str | None) -> str | None:
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
    return _infer_target_type(raw) or raw


def _infer_target_type(text: str | None) -> str | None:
    raw = str(text or "")
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
        for match in re.finditer(pattern, raw, re.IGNORECASE):
            matches.append((match.start(), match.end(), target_type))
    if not matches:
        return None
    matches.sort(key=lambda item: (item[1], item[0]))
    return matches[-1][2]


def _target_type_label(target_type: str | None) -> str:
    labels = {
        "person": "目标人",
        "cardboard box": "箱子",
        "chair": "椅子",
        "table": "桌子",
        "door": "门",
        "bag": "包",
        "phone": "手机",
    }
    return labels.get(target_type or "", target_type or "目标")


__all__ = [
    "map_steps_to_actions",
    "actions_from_legacy",
    "actions_to_legacy_dicts",
    "dedupe_actions",
    "build_action_reply",
    "format_stage_actions",
]
