import json
import logging
import time
import warnings

from flask import Response, request
warnings.filterwarnings("ignore", message=".*PydanticSerializationUnexpectedValue.*")
warnings.filterwarnings(
    "ignore", message="Pydantic serializer warnings:.*", category=UserWarning
)

logger = logging.getLogger("rag_agent.score")

RECURSION_LIMIT = 45
_app = None
_recursion_error = None
_timing_summary = None
_token_summary = None


def init():
    """
    Called once per worker process when the container starts. Builds the agent graph
    here (rather than at module import time) so its cost -- and any failure, e.g. an
    unreachable Cosmos account -- lands in the "Invoking user's init function" section
    of the deployment log instead of failing silently at import.
    """
    global _app, _recursion_error, _timing_summary, _token_summary
    started = time.perf_counter()
    from langgraph.errors import GraphRecursionError
    from graphs import plan_and_execute_app
    from helper_functions import timing_summary, token_summary

    _app = plan_and_execute_app
    _recursion_error = GraphRecursionError
    _timing_summary = timing_summary
    _token_summary = token_summary
    logger.info("init complete in %.1fs", time.perf_counter() - started)


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
    try:
        for stream_mode, chunk in _app.stream(
            inputs, config=config, stream_mode=["updates", "values"]
        ):
            if stream_mode == "updates":
                for node in chunk:
                    yield {
                        "type": "progress",
                        "node": node,
                        "elapsed": round(time.perf_counter() - started, 2),
                    }
            else:  # "values"
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
    _timing_summary(final_state, time.perf_counter() - started)
    _token_summary(final_state)
    yield {
        "type": "final",
        "response": final_state.get("response"),
        "elapsed": round(time.perf_counter() - started, 2),
    }


def run(raw_data):
    try:
        question = json.loads(raw_data)["question"]
    except (TypeError, ValueError, KeyError):
        return Response(
            json.dumps({"error": 'body must be {"question": "..."}'}),
            status=400,
            mimetype="application/json",
        )

    if "text/event-stream" in (request.headers.get("Accept") or ""):
        return Response(
            (
                f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
                for event in _run_agent(question)
            ),
            mimetype="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    for event in _run_agent(question):
        pass
    if event["type"] == "error":
        return Response(json.dumps(event), status=500, mimetype="application/json")
    return event
