"""本地 embedding 服务（OpenAI 兼容），给 RD-Agent 当 embedding 后端。

为什么需要它：
    RD-Agent 的 CoSTEER 在检索「相似的历史尝试/知识库」时必须调用 embedding。
    实测可用性：DeepSeek（无 embedding 产品）；智谱 `embedding-3` → **429 余额不足**；
    本机 ollama 的 runner 直接崩（`GGML_ASSERT(buf_dst) failed`，两个模型都崩）。
    → 用 fastembed（onnxruntime 后端，不拖 torch）+ 中文小模型，serve 一个 OpenAI 兼容端点。

跑法：
    HF_ENDPOINT=https://hf-mirror.com ~/.venvs/embed-shim/bin/python bridge/embed_shim.py
默认监听 127.0.0.1:11435，模型 BAAI/bge-small-zh-v1.5（512 维，首次启动自动下载 ~95MB）。

接到 RD-Agent（env，走 litellm 的 litellm_proxy 前缀，与智谱那套写法一致）：
    EMBEDDING_MODEL=litellm_proxy/bge-small-zh-v1.5
    LITELLM_PROXY_API_BASE=http://127.0.0.1:11435/v1
    LITELLM_PROXY_API_KEY=dummy
"""

from __future__ import annotations

import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MODEL = os.environ.get("EMBED_MODEL", "")
PORT = int(os.environ.get("EMBED_PORT", "11999"))
HOST = os.environ.get("EMBED_HOST", "127.0.0.1")

# 两个后端，按可用性自动选：
#   fastembed（onnxruntime，轻，但 Qdrant 镜像可能缺 model_optimized.onnx）
#   sentence-transformers（PyTorch，重，但标准 HF 仓库完整）—— 本机 /usr/bin/python3 自带
_BACKEND = os.environ.get("EMBED_BACKEND", "").lower()
if not _BACKEND:
    try:
        import fastembed  # noqa: F401

        _BACKEND = "fastembed"
    except ImportError:
        _BACKEND = "st"

if _BACKEND == "fastembed":
    from fastembed import TextEmbedding

    MODEL = MODEL or "BAAI/bge-small-zh-v1.5"
    print(f"[embed-shim] fastembed 加载 {MODEL} …", flush=True)
    _engine = TextEmbedding(model_name=MODEL)

    def _embed(texts: list[str]) -> list[list[float]]:
        return [] if not texts else [v.tolist() for v in _engine.embed(texts)]

else:
    from sentence_transformers import SentenceTransformer

    MODEL = MODEL or "BAAI/bge-small-zh-v1.5"
    print(f"[embed-shim] sentence-transformers 加载 {MODEL} …", flush=True)
    _engine = SentenceTransformer(MODEL)

    def _embed(texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return [v.tolist() for v in _engine.encode(texts, normalize_embeddings=True)]

_probe = _embed(["就绪探针"])[0]
print(
    f"[embed-shim] 就绪：后端={_BACKEND} 模型={MODEL} 维度={len(_probe)} "
    f"监听 http://{HOST}:{PORT}/v1",
    flush=True,
)


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _json(self, code: int, payload: dict) -> None:
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # /health、/v1/models
        if self.path.startswith("/v1/models"):
            self._json(200, {"object": "list", "data": [{"id": MODEL, "object": "model"}]})
        else:
            self._json(200, {"status": "ok", "model": MODEL})

    def do_POST(self) -> None:  # /v1/embeddings
        try:
            length = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(length) or b"{}")
            raw = body.get("input", "")
            texts = [raw] if isinstance(raw, str) else [str(t) for t in raw]
            vectors = _embed(texts)
            self._json(
                200,
                {
                    "object": "list",
                    "data": [
                        {"object": "embedding", "index": i, "embedding": vec}
                        for i, vec in enumerate(vectors)
                    ],
                    "model": MODEL,
                    "usage": {"prompt_tokens": 0, "total_tokens": 0},
                },
            )
        except Exception as exc:  # noqa: BLE001 - shim 不能把异常吞成 200
            self._json(500, {"error": {"message": f"{type(exc).__name__}: {exc}"}})

    def log_message(self, *args) -> None:  # 静音访问日志
        return


if __name__ == "__main__":
    ThreadingHTTPServer((HOST, PORT), Handler).serve_forever()
