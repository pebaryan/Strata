#!/usr/bin/env python3
"""Smoke-test that the OpenAI SSE endpoint emits tokens as the engine yields them."""
import http.client
import json
import pathlib
import sys
import threading
import time
from http.server import ThreadingHTTPServer

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from glm_http import Handler


class FakeTokenizer:
    def encode(self, _text, _add_special):
        return [1, 2]

    def decode(self, ids):
        return "".join(chr(96 + token) for token in ids)


class FakeRuntime:
    def __init__(self):
        self.lock = threading.Lock()
        self.tok = FakeTokenizer()
        self.eos = set()

    def chat_prompt(self, _req):
        return "test prompt"

    def generate_iter(self, _ids, _max_new):
        with self.lock:
            for token in (1, 2, 3):
                time.sleep(0.1)
                yield token


def main():
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.runtime = FakeRuntime()
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        conn = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=5)
        body = json.dumps({"messages": [{"role": "user", "content": "hi"}], "max_tokens": 3,
                           "stream": True, "stream_options": {"include_usage": True}})
        conn.request("POST", "/v1/chat/completions", body,
                     {"Content-Type": "application/json", "Accept": "text/event-stream"})
        response = conn.getresponse()
        assert response.status == 200
        arrivals = []
        usage = None
        while True:
            line = response.fp.readline().decode("utf-8")
            if not line:
                break
            if not line.startswith("data: "):
                continue
            payload = line[6:].strip()
            if payload == "[DONE]":
                break
            chunk = json.loads(payload)
            if chunk.get("choices") and chunk["choices"][0].get("finish_reason") is None:
                arrivals.append((time.monotonic(), chunk["choices"][0]["delta"]["content"]))
            if chunk.get("usage"):
                usage = chunk["usage"]
        assert [text for _, text in arrivals] == ["a", "b", "c"], arrivals
        assert all(arrivals[i + 1][0] - arrivals[i][0] >= 0.07 for i in range(2)), arrivals
        assert usage == {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5}, usage
        print("glm_http streaming smoke: PASS (incremental chunks and final usage)")
        conn.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


if __name__ == "__main__":
    main()
