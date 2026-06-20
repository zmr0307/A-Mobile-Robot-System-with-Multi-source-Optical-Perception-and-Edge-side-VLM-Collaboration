import json
import re

from wl100_demo.nlu_schema import (
    HANDOVER_OBJECTS,
    INTENTS,
    STEP_KINDS,
    WAYPOINTS,
    ParsedIntent,
    SemanticStep,
)
from wl100_demo.skill_mapper import actions_to_legacy_dicts, map_steps_to_actions


_EMPTY_WORDS = {"", "null", "none", "nil", "无", "没有", "空"}
_ACTION_TO_STEP = {
    "goto_a": SemanticStep("goto", "A"),
    "goto_b": SemanticStep("goto", "B"),
    "goto_c": SemanticStep("goto", "C"),
    "stage1": SemanticStep("find_person", "A"),
    "stage2": SemanticStep("find_person", "B"),
    "stage3": SemanticStep("find_person", "C"),
    "semantic_find_person": SemanticStep(
        "semantic_find_person", target_type="person"),
    "semantic_find_target": SemanticStep("semantic_find_target"),
    "home": SemanticStep("home", "HOME"),
    "patrol_all": SemanticStep("patrol_all"),
    "inspect_here": SemanticStep("inspect_here"),
    "generate_report": SemanticStep("generate_report"),
    "cancel": SemanticStep("cancel"),
}


def parse_vlm_plan_json(raw: str) -> ParsedIntent:
    """解析 VLM 输出，保留语义槽位。

    这里不做执行决策，也不直接发布 stage。解析失败只返回 chat，
    后续由 validator 判断是否需要本地补槽或追问。
    """
    cleaned = _clean_raw(raw)
    first = cleaned.find("{")
    last = cleaned.rfind("}")
    if first < 0 or last <= first:
        return ParsedIntent(
            intent="chat",
            steps=[],
            reply=cleaned[:200],
            raw=raw,
            parse_error="no_json",
        )
    try:
        obj = json.loads(cleaned[first:last + 1])
    except json.JSONDecodeError as exc:
        return ParsedIntent(
            intent="chat",
            steps=[],
            reply=cleaned[:200],
            raw=raw,
            parse_error=f"json:{exc}",
        )

    intent = _normalise_intent(obj.get("intent"))
    intent_as_step = None
    if intent in STEP_KINDS:
        intent_as_step = intent
        intent = "command"

    reply = str(obj.get("reply", "") or "").strip()
    steps = normalise_semantic_steps(obj.get("steps"))
    if not steps and intent_as_step:
        steps = [SemanticStep(intent_as_step)]
    if not steps:
        steps = normalise_legacy_actions(obj.get("actions"))
    if intent != "command":
        steps = []

    return ParsedIntent(intent=intent, steps=steps, reply=reply, raw=raw)


def parse_vlm_intent_json(raw: str):
    """兼容旧调用：返回 intent / stage actions / reply。"""
    parsed = parse_vlm_plan_json(raw)
    actions, _issues = map_steps_to_actions(parsed.steps)
    return parsed.intent, actions_to_legacy_dicts(actions), parsed.reply


def normalise_semantic_steps(raw_steps) -> list[SemanticStep]:
    if raw_steps is None:
        return []
    if isinstance(raw_steps, dict):
        raw_steps = [raw_steps]
    if not isinstance(raw_steps, list):
        return []

    steps: list[SemanticStep] = []
    for item in raw_steps:
        if not isinstance(item, dict):
            continue
        kind = str(item.get("kind") or "").strip().lower()
        if kind not in STEP_KINDS:
            continue
        waypoint = _normalise_waypoint(item.get("waypoint"))
        obj = _normalise_object(item.get("object"))
        target_description = _normalise_text(
            item.get("target_description")
            or item.get("person_description")
            or item.get("description")
            or item.get("target")
        )
        target_type = _normalise_target_type(
            item.get("target_type")
            or item.get("target_class")
            or item.get("type")
            or item.get("category")
            or item.get("object"),
            target_description,
        )
        if kind == "semantic_find_person":
            target_type = "person"
        elif kind == "semantic_find_target" and not target_type:
            target_type = _infer_target_type(target_description)
        steps.append(SemanticStep(
            kind=kind,
            waypoint=waypoint,
            object=obj,
            target_description=target_description,
            target_type=target_type,
        ))
    return steps


def normalise_legacy_actions(raw_actions) -> list[SemanticStep]:
    if raw_actions is None:
        return []
    if isinstance(raw_actions, str):
        raw_actions = [part.strip() for part in raw_actions.split(",")]
    if isinstance(raw_actions, dict):
        raw_actions = [raw_actions]
    if not isinstance(raw_actions, list):
        return []

    steps: list[SemanticStep] = []
    for item in raw_actions:
        target = None
        waypoint = None
        target_description = None
        target_type = None
        if isinstance(item, str):
            name = item.strip()
            if ":" in name:
                name, target = name.split(":", 1)
        elif isinstance(item, dict):
            name = (
                item.get("name")
                or item.get("stage")
                or item.get("action")
                or ""
            )
            target = item.get("target_object")
            waypoint = item.get("waypoint")
            target_description = (
                item.get("target_description")
                or item.get("person_description")
                or item.get("description")
                or item.get("target")
            )
            target_type = (
                item.get("target_type")
                or item.get("target_class")
                or item.get("type")
                or item.get("category")
            )
        else:
            continue
        name = str(name).strip()
        base = _ACTION_TO_STEP.get(name)
        if not base:
            continue
        obj = _normalise_object(target)
        desc = _normalise_text(target_description) or base.target_description
        norm_target_type = (
            _normalise_target_type(target_type or target, desc)
            or base.target_type
        )
        if base.kind == "semantic_find_person":
            norm_target_type = "person"
        elif base.kind == "semantic_find_target" and not norm_target_type:
            norm_target_type = _infer_target_type(desc)
        steps.append(SemanticStep(
            base.kind,
            _normalise_waypoint(waypoint) or base.waypoint,
            obj or base.object,
            desc,
            norm_target_type,
        ))
    return steps


def format_action_desc(actions) -> str:
    if not actions:
        return "[]"
    return "[" + ", ".join(
        f"{item.get('stage', '?')}"
        f"{':' + item['target_object'] if item.get('target_object') else ''}"
        for item in actions
    ) + "]"


def _clean_raw(raw: str) -> str:
    cleaned = re.sub(r"<think>.*?</think>", "", raw or "", flags=re.DOTALL)
    cleaned = re.sub(r"```json\s*", "", cleaned)
    cleaned = re.sub(r"```\s*", "", cleaned)
    return cleaned.strip()


def _normalise_intent(value) -> str:
    intent = str(value or "chat").strip()
    if intent in INTENTS or intent in STEP_KINDS:
        return intent
    return "chat"


def _normalise_waypoint(value):
    waypoint = str(value).strip().upper() if value is not None else ""
    if waypoint.lower() in _EMPTY_WORDS:
        return None
    return waypoint if waypoint in WAYPOINTS else None


def _normalise_object(value):
    obj = str(value).strip() if value is not None else ""
    if obj.lower() in _EMPTY_WORDS:
        return None
    return obj if obj in HANDOVER_OBJECTS else None


def _normalise_text(value):
    text = str(value).strip() if value is not None else ""
    if text.lower() in _EMPTY_WORDS:
        return None
    return text or None


def _normalise_target_type(value, description: str | None = None):
    raw = str(value).strip().lower() if value is not None else ""
    if raw in _EMPTY_WORDS:
        raw = ""
    aliases = {
        "person": "person",
        "people": "person",
        "human": "person",
        "人": "person",
        "人员": "person",
        "工作人员": "person",
        "员工": "person",
        "目标人": "person",
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
    return _infer_target_type(raw) or _infer_target_type(description)


def _infer_target_type(description: str | None):
    text = str(description or "")
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
        for match in re.finditer(pattern, text, re.IGNORECASE):
            matches.append((match.start(), match.end(), target_type))
    if not matches:
        return None
    matches.sort(key=lambda item: (item[1], item[0]))
    return matches[-1][2]
