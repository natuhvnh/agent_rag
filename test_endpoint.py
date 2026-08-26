"""
Send one question to the deployed agent and print the streamed result.

Usage:
    python3 test_endpoint.py
    python3 test_endpoint.py "your question here"

Override the target with ENDPOINT_URL (base URL, no /ask suffix).
"""

import json
import os
import sys

import requests
from dotenv import load_dotenv

load_dotenv()

DEFAULT_URL = "https://rag-agent.salmonground-3ab76589.uksouth.azurecontainerapps.io"
DEFAULT_QUESTION = "Search the web and give me the answer for question How Event Order API (POS) works ?"

url = os.environ.get("ENDPOINT_URL", DEFAULT_URL).rstrip("/") + "/ask"
question = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_QUESTION

api_key = os.environ.get("rag_api_key")
if not api_key:
    sys.exit("rag_api_key is not set (check .env or export it) -- the server will reject this request.")

headers = {
    "Content-Type": "application/json",
    "Accept": "text/event-stream",
    "x-api-key": api_key,
}
terminal = None
steps = 0
streamed_answer = ""

# stream=True disables buffering, equivalent to curl -N. The read timeout must exceed a
# cold-start agent run; ACA ingress caps a request at 240s regardless.
with requests.post(
    url, headers=headers, json={"question": question}, stream=True, timeout=(10, 300)
) as response:
    response.raise_for_status()
    for line in response.iter_lines():
        if not line:
            continue
        text = line.decode("utf-8")
        if not text.startswith("data: "):
            continue
        event = json.loads(text[len("data: "):])
        kind = event.get("type")
        if kind == "start":
            print(f"Q: {event['question']}\n")
        elif kind == "progress":
            # Interim plan/retrieval steps -- not printed, only counted.
            steps += 1
        elif kind == "token_reset":
            # A hallucination-check retry restarted the answer -- discard what was
            # printed so far rather than showing it concatenated with the retry.
            if streamed_answer:
                print("\n[answer restarted]\n")
            streamed_answer = ""
        elif kind == "token":
            streamed_answer += event["text"]
            print(event["text"], end="", flush=True)
        else:  # "final" or "error"
            terminal = event

# A stream ending without a terminal event is its own failure mode -- the run did not
# finish -- and is otherwise indistinguishable from success.
if terminal is None:
    print("\nSTREAM ENDED WITH NO TERMINAL EVENT -- the run did not complete.")
    sys.exit(1)

if terminal["type"] == "error":
    print(f"\nERROR: {terminal['message']}")
    sys.exit(1)

answer = terminal.get("response")
print(f"\n\n--- answer ({steps} steps, {terminal['elapsed']}s) ---")
if not streamed_answer:
    # Fell back to the non-streaming JSON path, or no tokens matched the filter --
    # print the final response so the answer is still visible either way.
    print(answer or "<empty response>")

# Measure rather than eyeball whether the answer was cut: a length that lands near a
# round token budget points at max_tokens, a short one points elsewhere.
print(f"\n--- diagnostics ---")
print(f"answer length : {len(answer or '')} chars (~{len((answer or '').split())} words)")
with open("last_response.json", "w") as fh:
    json.dump(terminal, fh, indent=2, ensure_ascii=False)
print("raw terminal event written to last_response.json")
