from dataclasses import dataclass, field
from typing import Optional


INTENTS = {"command", "query_history", "chat", "clarify"}
STEP_KINDS = {
    "goto", "find_person", "handover", "inspect_here",
    "semantic_find_person", "semantic_find_target",
    "home", "patrol_all", "generate_report", "cancel",
}
WAYPOINTS = {"A", "B", "C", "HOME"}
HANDOVER_OBJECTS = {"纸箱", "箱子"}
STAGE_NAMES = {
    "stage1", "stage2", "stage3",
    "goto_a", "goto_b", "goto_c",
    "semantic_find_person", "semantic_find_target",
    "home", "patrol_all", "inspect_here",
    "generate_report", "cancel",
}


@dataclass
class SemanticStep:
    kind: str
    waypoint: Optional[str] = None
    object: Optional[str] = None
    target_description: Optional[str] = None
    target_type: Optional[str] = None


@dataclass
class StageAction:
    stage: str
    target_object: Optional[str] = None
    waypoint: Optional[str] = None
    target_description: Optional[str] = None
    target_type: Optional[str] = None

    def as_dict(self):
        data = {
            "stage": self.stage,
            "target_object": self.target_object,
        }
        if self.waypoint:
            data["waypoint"] = self.waypoint
        if self.target_description:
            data["target_description"] = self.target_description
        if self.target_type:
            data["target_type"] = self.target_type
        return data


@dataclass
class ParsedIntent:
    intent: str = "chat"
    steps: list[SemanticStep] = field(default_factory=list)
    reply: str = ""
    raw: str = ""
    parse_error: str = ""


@dataclass
class PlannerResult:
    parsed: ParsedIntent
    raw: str
    latency_sec: float = 0.0


@dataclass
class Decision:
    intent: str
    reply: str
    actions: list[StageAction] = field(default_factory=list)
    executable: bool = False
    reason: str = ""
    repaired_from: str = ""

    @property
    def observe_detail(self) -> str:
        action_text = format_stage_actions(self.actions)
        repair = f"; repaired_from={self.repaired_from}" if self.repaired_from else ""
        return (
            f"intent={self.intent}; actions={action_text}{repair}; "
            f"reason={self.reason or 'ok'}; reply={self.reply or '（空）'}"
        )


def format_stage_actions(actions: list[StageAction]) -> str:
    if not actions:
        return "[]"
    return "[" + ", ".join(
        f"{item.stage}{':' + item.target_object if item.target_object else ''}"
        f"{'@' + item.waypoint if item.waypoint else ''}"
        f"{'<' + item.target_type + '>' if item.target_type else ''}"
        f"{'(' + item.target_description + ')' if item.target_description else ''}"
        for item in actions
    ) + "]"
