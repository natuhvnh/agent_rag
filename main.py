"""
Azure AI Foundry hosted-agent entry point (Responses protocol).

Deployed via deploy_azure_foundry.py. The runtime uploads this repo as a code zip and runs
`python main.py`, which must bind 0.0.0.0:${PORT:-8088} and serve the responses protocol --
ResponsesAgentServerHost (azure-ai-agentserver-responses) provides that server, including the
/readiness probe.
"""

import asyncio
import os
import time
from pathlib import Path

os.chdir(Path(__file__).parent)

from dotenv import load_dotenv

load_dotenv()

import warnings

warnings.filterwarnings("ignore", message=".*PydanticSerializationUnexpectedValue.*")
warnings.filterwarnings(
    "ignore", message="Pydantic serializer warnings:.*", category=UserWarning
)

from azure.ai.agentserver.responses import (
    CreateResponse,
    ResponseContext,
    ResponsesAgentServerHost,
    TextResponse,
)
from langgraph.errors import GraphRecursionError

from graphs import plan_and_execute_app

RECURSION_LIMIT = 45

app = ResponsesAgentServerHost()


async def _answer(question):
    """
    Run the plan-and-execute agent and yield the final answer text once, buffering the
    streamed answer tokens along the way.
    """
    loop = asyncio.get_event_loop()
    queue = asyncio.Queue()

    def produce():
        # The sentinel must be enqueued unconditionally: the consumer below blocks on
        # queue.get() forever without it, and this runs in an executor whose Future is
        # discarded, so an escaping exception would be lost.
        try:
            inputs = {
                "question": question,
                "past_steps": [],
                "aggregated_context": "",
                "tool": "",
            }
            config = {"recursion_limit": RECURSION_LIMIT}
            final_state = None
            try:
                for ns, stream_mode, chunk in plan_and_execute_app.stream(
                    inputs,
                    config=config,
                    stream_mode=["updates", "values", "messages"],
                    subgraphs=True,
                ):
                    if ns:
                        if stream_mode == "messages":
                            msg, metadata = chunk
                            if (
                                ns[0].startswith("get_final_answer:")
                                and metadata.get("langgraph_node")
                                == "answer_question_from_context"
                                and msg.content
                            ):
                                loop.call_soon_threadsafe(
                                    queue.put_nowait, ("token", msg.id, msg.content)
                                )
                        continue
                    if stream_mode == "values":
                        final_state = chunk
            except GraphRecursionError:
                loop.call_soon_threadsafe(
                    queue.put_nowait,
                    ("final", None, "The answer wasn't found in the data after recursion limit"),
                )
                return
            final_state = final_state or {}
            loop.call_soon_threadsafe(
                queue.put_nowait, ("final", None, final_state.get("response"))
            )
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, None)

    loop.run_in_executor(None, produce)

    buffer = ""
    last_msg_id = None
    while True:
        item = await queue.get()
        if item is None:
            break
        kind, msg_id, text = item
        if kind == "token":
            if msg_id != last_msg_id:
                # A new message id means a fresh answer generation started (e.g. a
                # hallucination-check retry) -- discard whatever was buffered so far.
                last_msg_id = msg_id
                buffer = ""
            buffer += text
        else:  # "final"
            if not buffer:
                # No tokens matched the filter (or the recursion limit was hit) -- fall
                # back to the terminal response text.
                buffer = text or ""

    yield buffer


@app.response_handler
async def handle(
    request: CreateResponse,
    context: ResponseContext,
    _cancellation_signal: asyncio.Event,
):
    question = await context.get_input_text() or ""
    return TextResponse(context, request, text=_answer(question))


if __name__ == "__main__":
    app.run()
