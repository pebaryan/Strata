#!/usr/bin/env python3
"""Persistent OpenAI-compatible HTTP front end for the Strata GLM engine."""
from __future__ import annotations
import argparse, json, pathlib, subprocess, sys, threading, time, uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from strata_tokenizer import Tokenizer
from jinja2 import Environment


class Runtime:
    def __init__(self, a):
        td = pathlib.Path(a.tokenizer)
        vocab = json.loads((td / "vocab.json").read_text(encoding="utf-8"))
        tokens = [""] * len(vocab)
        for tok, i in vocab.items(): tokens[i] = tok
        merges = (td / "merges.txt").read_text(encoding="utf-8").splitlines()
        types = json.loads((td / "token_type.json").read_text(encoding="utf-8"))
        cfg = json.loads((td / "tokenizer.json").read_text(encoding="utf-8"))
        self.tok = Tokenizer(tokens, merges, types, cfg.get("pre", "glm4"), cfg.get("special_ids", {}))
        self.template = Environment(autoescape=False, extensions=["jinja2.ext.loopcontrols"]).from_string(
            (td / "chat_template.jinja").read_text(encoding="utf-8"))
        self.eos = {int(cfg["special_ids"].get(k, -1)) for k in
                    ("tokenizer.ggml.eos_token_id", "tokenizer.ggml.eot_token_id")}
        self.eos.discard(-1)
        self.lock = threading.Lock()
        cmd = [a.binary, a.pack, a.shard, "--serve", a.output, a.norm]
        self.proc = subprocess.Popen(cmd, cwd=a.cwd, text=True, bufsize=1,
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        threading.Thread(target=self._drain_err, daemon=True).start()
        deadline = time.time() + a.startup_timeout
        while time.time() < deadline:
            line = self.proc.stdout.readline()
            if not line: raise RuntimeError("Strata GLM exited during startup")
            if line.startswith("READY "): break
        else: raise TimeoutError("Strata GLM did not become ready")

    def _drain_err(self):
        for line in self.proc.stderr: sys.stderr.write(line)

    def chat_prompt(self, req):
        return self.template.render(messages=req.get("messages", []), tools=req.get("tools", []),
                                    add_generation_prompt=True,
                                    reasoning_effort=req.get("reasoning_effort", "max"), clear_thinking=False)

    def generate(self, ids, max_new):
        if not ids: raise ValueError("prompt tokenizes to an empty sequence")
        with self.lock:
            self.proc.stdin.write("GEN %d %s\n" % (max_new, ",".join(map(str, ids))))
            self.proc.stdin.flush(); out = []
            while True:
                line = self.proc.stdout.readline()
                if not line: raise RuntimeError("Strata GLM engine exited")
                line = line.strip()
                if line.startswith("T "): out.append(int(line[2:]))
                elif line.startswith("DONE "): return out
                elif line.startswith("ERR "): raise RuntimeError(line[4:])


class Handler(BaseHTTPRequestHandler):
    server_version = "strata-glm/1.0"
    def send_json(self, code, obj):
        data=json.dumps(obj,ensure_ascii=False).encode(); self.send_response(code)
        self.send_header("Content-Type","application/json"); self.send_header("Content-Length",str(len(data)))
        self.end_headers(); self.wfile.write(data)
    def body(self): return json.loads(self.rfile.read(int(self.headers.get("Content-Length","0"))) or b"{}")
    def do_GET(self):
        if self.path in ("/health","/ready"):
            return self.send_json(200,{"status":"ok","engine":"strata","model":"glm-5.3-flash"})
        if self.path == "/v1/models":
            return self.send_json(200,{"object":"list","data":[{"id":"glm-5.3-flash","object":"model","owned_by":"strata"}]})
        if self.path == "/props":
            return self.send_json(200,{"model_path":"glm-5.3-flash","total_slots":1,"chat_template":"glm5next"})
        self.send_json(404,{"error":{"message":"not found","type":"invalid_request_error"}})
    def do_POST(self):
        try:
            req=self.body(); rt=self.server.runtime
            if self.path == "/tokenize":
                return self.send_json(200,{"tokens":rt.tok.encode(req.get("content",req.get("prompt","")),True)})
            if self.path == "/detokenize":
                return self.send_json(200,{"content":rt.tok.decode(req.get("tokens",[]))})
            chat=self.path == "/v1/chat/completions"
            if not chat and self.path not in ("/v1/completions","/completion"):
                return self.send_json(404,{"error":{"message":"not found","type":"invalid_request_error"}})
            prompt=rt.chat_prompt(req) if chat else req.get("prompt",req.get("content",""))
            if isinstance(prompt,list): prompt=prompt[0]
            ids=rt.tok.encode(str(prompt),True); max_new=int(req.get("max_tokens",req.get("n_predict",128)))
            max_new=max(1,min(max_new,8192-len(ids))); out=rt.generate(ids,max_new); text=rt.tok.decode(out)
            if self.path == "/completion":
                return self.send_json(200,{"content":text,"tokens_predicted":len(out),"stop":bool(out and out[-1] in rt.eos)})
            now=int(time.time()); rid=("chatcmpl-" if chat else "cmpl-")+uuid.uuid4().hex
            if req.get("stream"):
                self.send_response(200); self.send_header("Content-Type","text/event-stream")
                self.send_header("Cache-Control","no-cache"); self.end_headers()
                for token in out:
                    ch={"id":rid,"object":"chat.completion.chunk" if chat else "text_completion",
                        "created":now,"model":req.get("model","glm-5.3-flash"),"choices":[{"index":0,
                        "delta":{"content":rt.tok.decode([token])} if chat else {},
                        "text":"" if chat else rt.tok.decode([token]),"finish_reason":None}]}
                    self.wfile.write(("data: "+json.dumps(ch,ensure_ascii=False)+"\n\n").encode()); self.wfile.flush()
                fin={"id":rid,"object":"chat.completion.chunk" if chat else "text_completion","created":now,
                     "model":req.get("model","glm-5.3-flash"),"choices":[{"index":0,"delta":{} if chat else None,
                     "text":"","finish_reason":"stop" if out and out[-1] in rt.eos else "length"}]}
                self.wfile.write(("data: "+json.dumps(fin,ensure_ascii=False)+"\n\ndata: [DONE]\n\n").encode()); return
            choice={"index":0,"finish_reason":"stop" if out and out[-1] in rt.eos else "length"}
            if chat: choice["message"]={"role":"assistant","content":text}
            else: choice["text"]=text
            self.send_json(200,{"id":rid,"object":"chat.completion" if chat else "text_completion",
                                "created":now,"model":req.get("model","glm-5.3-flash"),"choices":[choice],
                                "usage":{"prompt_tokens":len(ids),"completion_tokens":len(out),"total_tokens":len(ids)+len(out)}})
        except Exception as e:
            self.send_json(500,{"error":{"message":str(e),"type":"engine_error"}})
    def log_message(self, fmt, *args): sys.stderr.write("http: "+fmt%args+"\n")


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--cwd",default=str(pathlib.Path(__file__).resolve().parents[1]))
    ap.add_argument("--binary",default="build-glm5/strata-glm"); ap.add_argument("--pack",required=True)
    ap.add_argument("--shard",required=True); ap.add_argument("--tokenizer",required=True)
    ap.add_argument("--output",default="/home/peb/moredata/glm5-head-gate/w_output.bin")
    ap.add_argument("--norm",default="/home/peb/moredata/glm5-head-gate/w_output_norm.bin")
    ap.add_argument("--host",default="0.0.0.0"); ap.add_argument("--port",type=int,default=8080)
    ap.add_argument("--startup-timeout",type=int,default=300); a=ap.parse_args()
    runtime=Runtime(a); server=ThreadingHTTPServer((a.host,a.port),Handler); server.runtime=runtime
    print("strata GLM OpenAI server listening on %s:%d"%(a.host,a.port),flush=True)
    try: server.serve_forever()
    finally:
        if runtime.proc.poll() is None: runtime.proc.stdin.write("QUIT\n"); runtime.proc.stdin.flush()
if __name__ == "__main__": main()
