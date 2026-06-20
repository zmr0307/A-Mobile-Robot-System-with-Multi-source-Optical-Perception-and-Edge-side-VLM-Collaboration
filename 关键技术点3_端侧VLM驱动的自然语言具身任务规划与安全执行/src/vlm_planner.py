from wl100_demo.nlu_schema import PlannerResult
from wl100_demo.vlm_client import VlmClient
from wl100_demo.vlm_intent_parser import parse_vlm_plan_json


class VlmPlanner:
    def __init__(self, system_prompt: str, client: VlmClient):
        self.system_prompt = system_prompt
        self.client = client

    def plan(self, text: str) -> PlannerResult:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": text},
        ]
        raw, latency = self.client.chat(messages)
        parsed = parse_vlm_plan_json(raw)
        return PlannerResult(
            parsed=parsed,
            raw=raw,
            latency_sec=latency,
        )
