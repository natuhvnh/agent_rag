from dotenv import load_dotenv
import os
import time
import langgraph
from helper_functions import text_wrap, timing_summary
from graphs import plan_and_execute_app

load_dotenv()
# --- Set environment variable for debugging (optional) ---
os.environ["PYDEVD_WARN_EVALUATION_TIMEOUT"] = "100000"
# Work around a duplicate-OpenMP-runtime conflict: torch (pulled in transitively via
# langchain_community.vectorstores.FAISS -> transformers) and faiss-cpu each bundle their own
# libomp.dylib on macOS, which otherwise makes the first real FAISS search abort.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import warnings

warnings.filterwarnings("ignore", message=".*PydanticSerializationUnexpectedValue.*")
warnings.filterwarnings(
    "ignore", message="Pydantic serializer warnings:.*", category=UserWarning
)


def execute_plan_and_print_steps(inputs, recursion_limit=45):
    """
    Executes the plan-and-execute agent workflow and prints each step.

    Args:
        inputs (dict): The initial input state for the plan-and-execute agent.
        recursion_limit (int): Maximum number of steps to prevent infinite loops.

    Returns:
        tuple: (response, final_state)
            response (str): The final answer or message if not found.
            final_state (dict): The final state after execution.
    """
    _run_start = time.perf_counter()
    final_state = None
    config = {"recursion_limit": recursion_limit}
    try:
        for stream_mode, chunk in plan_and_execute_app.stream(
            inputs, config=config, stream_mode=["updates", "values"]
        ):
            if stream_mode == "updates":
                for _, agent_state_value in chunk.items():
                    print(f" curr step: {agent_state_value}")
            else:  # "values"
                final_state = chunk
        response = final_state["response"]
    except langgraph.errors.GraphRecursionError:
        response = "The answer wasn't found in the data."
    print(text_wrap(f" the final answer is: {response}"))
    _total_runtime = time.perf_counter() - _run_start
    if final_state is not None:
        timing_summary(final_state, _total_runtime)
    return response, final_state


#
if __name__ == "__main__":
    initial_input = {
        "question": "What is Hogwarts?",
        "past_steps": [],
        "aggregated_context": "",
        "tool": "",
    }
    final_answer, final_state = execute_plan_and_print_steps(initial_input)
    print("=" * 50)
    print(final_answer)