"""Baseline server: Hugging Face transformers `generate`, one request at a time, behind the same
OpenAI-compatible interface the judges use. No batching and no constrained decoding (transformers has none
built in); JSON is requested by the system prompts only, and Markdown code fences are stripped from replies.
"""
import argparse
import base64
import io
import json
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

FENCE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.S)


def hf_messages(messages):
    out = []
    for m in messages:
        content = m["content"]
        if isinstance(content, str):
            out.append({"role": m["role"], "content": [{"type": "text", "text": content}]})
            continue
        parts = []
        for part in content:
            if part["type"] == "text":
                parts.append({"type": "text", "text": part["text"]})
            else:
                data = base64.b64decode(part["image_url"]["url"].split(",", 1)[1])
                parts.append({"type": "image", "image": Image.open(io.BytesIO(data)).convert("RGB")})
        out.append({"role": m["role"], "content": parts})
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    processor = AutoProcessor.from_pretrained(args.model)
    model = Qwen3VLForConditionalGeneration.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    lock = threading.Lock()
    counters = {"prompt": 0, "generation": 0, "running": 0, "waiting": 0}

    def generate(body):
        inputs = processor.apply_chat_template(hf_messages(body["messages"]), tokenize=True,
                                               add_generation_prompt=True, return_dict=True,
                                               return_tensors="pt").to(model.device)
        max_tokens = body.get("max_tokens", 1024)
        with torch.inference_mode():
            output = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
        new = output[0, inputs["input_ids"].shape[1]:]
        text = processor.decode(new, skip_special_tokens=True)
        match = FENCE.match(text)
        text = match.group(1) if match else text
        finish = "length" if len(new) >= max_tokens else "stop"
        return text, inputs["input_ids"].shape[1], len(new), finish

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def reply(self, payload, content_type="application/json"):
            data = payload.encode() if isinstance(payload, str) else json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path.startswith("/metrics"):
                self.reply(f"vllm:prompt_tokens_total {counters['prompt']}\n"
                           f"vllm:generation_tokens_total {counters['generation']}\n"
                           f"vllm:num_requests_running {counters['running']}\n"
                           f"vllm:num_requests_waiting {counters['waiting']}\n", "text/plain")
            else:
                self.reply({"object": "list", "data": [{"id": "judge", "object": "model"}]})

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            counters["waiting"] += 1
            with lock:
                counters["waiting"] -= 1
                counters["running"] = 1
                text, n_prompt, n_new, finish = generate(body)
                counters["running"] = 0
            counters["prompt"] += n_prompt
            counters["generation"] += n_new
            self.reply({"id": f"hf-{time.time_ns()}", "object": "chat.completion", "created": int(time.time()),
                        "model": "judge", "choices": [{"index": 0, "finish_reason": finish,
                                                       "message": {"role": "assistant", "content": text}}],
                        "usage": {"prompt_tokens": n_prompt, "completion_tokens": n_new,
                                  "total_tokens": n_prompt + n_new}})

    print(f"serving {args.model} on port {args.port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
