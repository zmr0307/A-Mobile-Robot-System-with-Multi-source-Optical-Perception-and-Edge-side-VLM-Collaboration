import json
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class VlmClientError(RuntimeError):
    pass


class VlmClient:
    def __init__(self, api_url: str, timeout: float,
                 temperature: float = 0.4, max_tokens: int = 2048):
        self.api_url = api_url
        self.timeout = float(timeout)
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)

    def chat(self, messages):
        payload = {
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        req = Request(
            self.api_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        t0 = time.time()
        try:
            with urlopen(req, timeout=self.timeout) as resp:
                resp_json = json.loads(resp.read().decode("utf-8"))
        except (HTTPError, URLError, TimeoutError) as exc:
            raise VlmClientError(str(exc)) from exc
        try:
            raw = resp_json["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise VlmClientError("VLM 响应格式异常") from exc
        return raw, time.time() - t0

