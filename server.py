"""
FastAPI HTTP wrapper for the plan-and-execute RAG agent, for deployment on Azure
Container Apps.
"""

import asyncio
import json
import logging
import os
import secrets
import time
import warnings

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

warnings.filterwarnings("ignore", message=".*PydanticSerializationUnexpectedValue.*")
warnings.filterwarnings(
    "ignore", message="Pydantic serializer warnings:.*", category=UserWarning
)

logger = logging.getLogger("rag_agent.server")

RECURSION_LIMIT = 45

# Loaded here (not just transitively via dependencies.py, which is imported later inside
# _load()) so a missing key fails at import time. In ACA the value arrives as a real env
# var and load_dotenv() is a no-op.
load_dotenv()
API_KEY = os.environ["rag_api_key"]
if not API_KEY:
    # An empty string passes os.environ[...] but would make compare_digest("", "") true
    # for a request with no header at all -- i.e. /ask silently unauthenticated.
    raise RuntimeError("rag_api_key is empty -- refusing to start with /ask unauthenticated.")
# Compared as bytes below: Starlette decodes header bytes as latin-1, so a non-ASCII
# x-api-key value would otherwise raise TypeError out of compare_digest (str, str) --
# an unhandled 500 on an unauthenticated path instead of a clean 401.
API_KEY_BYTES = API_KEY.encode()


def require_api_key(x_api_key: str = Header(None)):
    # compare_digest, not ==, so response time doesn't leak the key prefix.
    if not secrets.compare_digest((x_api_key or "").encode(), API_KEY_BYTES):
        raise HTTPException(status_code=401, detail="Invalid or missing x-api-key")

_app_graph = None
_recursion_error = None
_timing_summary = None
_token_summary = None

app = FastAPI()


class AskRequest(BaseModel):
    question: str


@app.on_event("startup")
def _load():
    """
    Build the agent graph here (rather than at module import time) so its cost --
    and any failure, e.g. an unreachable Cosmos account -- lands in the ACA startup
    logs instead of failing silently at import.
    """
    global _app_graph, _recursion_error, _timing_summary, _token_summary
    started = time.perf_counter()
    from langgraph.errors import GraphRecursionError
    from graphs import plan_and_execute_app
    from helper_functions import timing_summary, token_summary

    _app_graph = plan_and_execute_app
    _recursion_error = GraphRecursionError
    _timing_summary = timing_summary
    _token_summary = token_summary
    logger.info("startup complete in %.1fs", time.perf_counter() - started)


@app.get("/healthz")
def healthz():
    if _app_graph is None:
        return JSONResponse({"status": "starting"}, status_code=503)
    return {"status": "ok"}


def _run_agent(question):
    """
    Run the plan-and-execute agent and yield one event dict per step, ending with
    exactly one terminal event of type "final" or "error". A stream that ends without
    a terminal event was truncated (e.g. the worker was killed mid-run).
    """
    started = time.perf_counter()
    inputs = {
        "question": question,
        "past_steps": [],
        "aggregated_context": "",
        "tool": "",
    }
    config = {"recursion_limit": RECURSION_LIMIT}
    final_state = None

    yield {"type": "start", "question": question}
    last_msg_id = None
    try:
        for ns, stream_mode, chunk in _app_graph.stream(
            inputs, config=config, stream_mode=["updates", "values", "messages"], subgraphs=True
        ):
            if ns:
                # A chunk from inside a sub-graph. Only the final-answer sub-graph's
                # own token stream is surfaced to the client; its "updates"/"values"
                # chunks have a different (QualitativeAnswerGraphState) shape and must
                # not be treated as top-level progress or clobber final_state below.
                if stream_mode == "messages":
                    msg, metadata = chunk
                    if (
                        ns[0].startswith("get_final_answer:")
                        and metadata.get("langgraph_node") == "answer_question_from_context"
                        and msg.content
                    ):
                        if msg.id != last_msg_id:
                            # A new message id means a fresh answer generation started
                            # (e.g. a hallucination-check retry) -- tell the client to
                            # discard whatever it had buffered so far.
                            last_msg_id = msg.id
                            yield {
                                "type": "token_reset",
                                "elapsed": round(time.perf_counter() - started, 2),
                            }
                        yield {
                            "type": "token",
                            "text": msg.content,
                            "elapsed": round(time.perf_counter() - started, 2),
                        }
                continue
            if stream_mode == "updates":
                for node in chunk:
                    yield {
                        "type": "progress",
                        "node": node,
                        "elapsed": round(time.perf_counter() - started, 2),
                    }
            elif stream_mode == "values":
                final_state = chunk
    except _recursion_error:
        yield {
            "type": "final",
            "response": "The answer wasn't found in the data after recursion limit",
            "recursion_limit_hit": True,
            "elapsed": round(time.perf_counter() - started, 2),
        }
        return
    except Exception:
        logger.exception("agent run failed")
        yield {
            "type": "error",
            "message": "Agent run failed. See deployment logs.",
            "elapsed": round(time.perf_counter() - started, 2),
        }
        return

    final_state = final_state or {}
    # Diagnostics only. These index record fields unguarded, so a malformed
    # node_timings/node_tokens entry raises -- which previously escaped this generator
    # and stalled the SSE stream before the terminal event below was ever emitted.
    try:
        _timing_summary(final_state, time.perf_counter() - started)
        _token_summary(final_state)
    except Exception:
        logger.exception("summary reporting failed")
    yield {
        "type": "final",
        "response": final_state.get("response"),
        "elapsed": round(time.perf_counter() - started, 2),
    }


@app.post("/ask", dependencies=[Depends(require_api_key)])
async def ask(payload: AskRequest, request: Request):
    question = payload.question

    if "text/event-stream" in (request.headers.get("accept") or ""):
        async def sse():
            queue = asyncio.Queue()
            loop = asyncio.get_event_loop()

            def produce():
                # The sentinel must be enqueued unconditionally: the consumer below
                # blocks on queue.get() forever without it, and this runs in an executor
                # whose Future is discarded, so an escaping exception would be lost.
                try:
                    for event in _run_agent(question):
                        loop.call_soon_threadsafe(queue.put_nowait, event)
                except Exception:
                    logger.exception("SSE producer failed")
                finally:
                    loop.call_soon_threadsafe(queue.put_nowait, None)

            loop.run_in_executor(None, produce)
            while True:
                event = await queue.get()
                if event is None:
                    break
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

        return StreamingResponse(
            sse(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    event = await asyncio.get_event_loop().run_in_executor(
        None, lambda: list(_run_agent(question))[-1]
    )
    if event["type"] == "error":
        return JSONResponse(event, status_code=500)
    return event
