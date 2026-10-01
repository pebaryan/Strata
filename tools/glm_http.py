#!/usr/bin/env python3
"""Small HTTP adapter for the Strata GLM engine.

POST /generate with {"input_dump": "/path/to/hc_init.bin"}; the dump is the
engine's 5-token HC input format.  The adapter keeps the HTTP process alive
and returns the engine's token/result as JSON.
"""
import argparse, json, os, re, subprocess
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

class Handler(BaseHTTPRequestHandler):
    server_version = "strata-glm-http/1.0"
    def _send(self, code, obj):
        data = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers(); self.wfile.write(data)
    def do_GET(self):
        if self.path == "/health" or self.path == "/ready":
            self._send(200, {"status":"ok", "engine":"strata-glm", "branch":self.server.branch})
        else: self._send(404, {"error":"not found"})
    def do_POST(self):
        if self.path != "/generate": self._send(404, {"error":"not found"}); return
        try:
            n = int(self.headers.get("Content-Length", "0")); req = json.loads(self.rfile.read(n) or b"{}")
            inp = req.get("input_dump", self.server.default_input)
            if not inp or not os.path.isfile(inp): raise ValueError("input_dump does not exist")
            p = subprocess.run([self.server.binary, self.server.pack, self.server.shard, inp],
                               text=True, capture_output=True, timeout=self.server.timeout)
            m = re.search(r"argmax = ([-0-9]+)\s+\(logit ([^)]*)\)", p.stdout)
            if p.returncode != 0 or not m:
                self._send(500, {"error":"engine failed", "returncode":p.returncode,
                                 "stderr":p.stderr[-4000:], "stdout":p.stdout[-4000:]}); return
            self._send(200, {"token":int(m.group(1)), "logit":float(m.group(2)), "engine":"strata-glm"})
        except Exception as e: self._send(400, {"error":str(e)})
    def log_message(self, fmt, *args): pass

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--binary", default="build-glm5/strata-glm")
    ap.add_argument("--pack", required=True); ap.add_argument("--shard", required=True)
    ap.add_argument("--input", default="/home/peb/moredata/glm5-oracle-full/hc_init.bin")
    ap.add_argument("--port", type=int, default=8080); ap.add_argument("--timeout", type=int, default=300)
    a=ap.parse_args(); s=ThreadingHTTPServer(("0.0.0.0",a.port),Handler)
    s.binary=a.binary; s.pack=a.pack; s.shard=a.shard; s.default_input=a.input; s.timeout=a.timeout; s.branch="glm5next-port"
    print(f"strata-glm HTTP listening on :{a.port}", flush=True); s.serve_forever()
if __name__ == "__main__": main()
