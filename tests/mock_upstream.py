"""Fake OpenAI-compatible backend for tests/e2e.sh: answers "answered-by-<name>"."""
import json, sys, time, uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
NAME = sys.argv[1]; PORT = int(sys.argv[2]); app = FastAPI()
@app.post("/v1/chat/completions")
async def cc(req: Request):
    body = await req.json(); text = f"answered-by-{NAME} model={body.get('model')}"
    if body.get("stream"):
        def gen():
            for w in text.split(" "):
                yield "data: " + json.dumps({"id":"x","object":"chat.completion.chunk","created":0,"model":body.get("model"),"choices":[{"index":0,"delta":{"content":w+" "},"finish_reason":None}]}) + "\n\n"
            yield "data: " + json.dumps({"id":"x","object":"chat.completion.chunk","created":0,"model":body.get("model"),"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}) + "\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(gen(), media_type="text/event-stream")
    return JSONResponse({"id":"x","object":"chat.completion","created":0,"model":body.get("model"),"choices":[{"index":0,"message":{"role":"assistant","content":text},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}})
uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="warning")
