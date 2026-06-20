import re


_ACTION_START = (
    r"(?:去|到|前往|开到|回|返回|归位|找|寻找|寻|看|看看|看一下|"
    r"观察|查看|描述|送|交接|递|生成|出|导出|取消|停止|"
    r"原地|巡检|[a-cA-C](?:点|观测点|那边|那里)?)"
)

_TASK_CONNECTOR_RE = re.compile(
    rf"(然后|接着|随后|之后|再|最后|找到以后|找到后|完成后|"
    rf"完了之后|并|以及|同时)(?=\s*{_ACTION_START})"
)

_LEADING_CONNECTOR_RE = re.compile(
    r"^(先|首先|然后|接着|随后|之后|再|最后|找到以后|找到后|"
    r"完成后|完了之后|并|以及|同时)\s*"
)

_TRAILING_PUNCT_RE = re.compile(r"[，,。；;、\s]+$")


def segment_tasks(text: str) -> list[str]:
    """Conservatively split a long command into executable task clauses.

    This deliberately avoids splitting on ordinary commas, so descriptions like
    "拿着箱子、穿着黑色裤子、坐着的人" stay as one semantic target.
    """
    source = _normalise(text)
    if not source:
        return []

    marked = _TASK_CONNECTOR_RE.sub("|", source)
    parts = [_clean_segment(part) for part in marked.split("|")]
    return [part for part in parts if part]


def should_segment(text: str) -> bool:
    return len(segment_tasks(text)) >= 2


def _normalise(text: str) -> str:
    text = (text or "").strip()
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text


def _clean_segment(text: str) -> str:
    text = (text or "").strip()
    text = _LEADING_CONNECTOR_RE.sub("", text)
    text = _TRAILING_PUNCT_RE.sub("", text)
    return text.strip()
