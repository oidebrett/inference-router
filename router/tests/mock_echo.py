"""Fake OpenAI-compatible backend for tests/e2e_masked.sh.

Appends every request body it receives to RECORD (so the test can prove no real
value reached it) and echoes the last user message back. If the message contains
"TOOL", it answers with a send_email tool call that copies the first placeholder.
Streams in 3-character pieces so placeholders are split across chunks."""
import json, re, sys, uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

PORT, RECORD = int(sys.argv[1]), sys.argv[2]
app = FastAPI()

@app.get("/v1/models")
async def models():
    return {"object": "list", "data": [{"id": "echo", "object": "model"}]}

@app.post("/v1/chat/completions")
async def cc(req: Request):
    body = await req.json()
    with open(RECORD, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(body) + "\n")
    last = next((m for m in reversed(body["messages"]) if m["role"] == "user"), {"content": ""})
    content = last["content"] if isinstance(last["content"], str) else json.dumps(last["content"])
    token = (re.findall(r"<[A-Z_]+_[0-9a-f]{8,16}>", content) or ["none"])[0]
    tool = "TOOL" in content
    text = f"seen: {content}"
    args = json.dumps({"to": token})
    base = {"id": "x", "created": 0, "model": body.get("model")}
    if body.get("stream"):
        def gen():
            if tool:
                for i in range(0, len(args), 3):
                    call = {"index": 0, "function": {"arguments": args[i:i+3]}}
                    if i == 0:
                        call.update(id="call_1", type="function", function={"name": "send_email", "arguments": args[:3]})
                    yield "data: " + json.dumps({**base, "object": "chat.completion.chunk", "choices": [
                        {"index": 0, "delta": {"tool_calls": [call]}, "finish_reason": None}]}) + "\n\n"
                finish = "tool_calls"
            else:
                for i in range(0, len(text), 3):
                    yield "data: " + json.dumps({**base, "object": "chat.completion.chunk", "choices": [
                        {"index": 0, "delta": {"content": text[i:i+3]}, "finish_reason": None}]}) + "\n\n"
                finish = "stop"
            yield "data: " + json.dumps({**base, "object": "chat.completion.chunk", "choices": [
                {"index": 0, "delta": {}, "finish_reason": finish}]}) + "\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(gen(), media_type="text/event-stream")
    message = {"role": "assistant", "content": text}
    if tool:
        message = {"role": "assistant", "content": None, "tool_calls": [
            {"id": "call_1", "type": "function", "function": {"name": "send_email", "arguments": args}}]}
    return JSONResponse({**base, "object": "chat.completion", "choices": [
        {"index": 0, "message": message, "finish_reason": "tool_calls" if tool else "stop"}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}})

uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="warning")
