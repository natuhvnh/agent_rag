from dotenv import load_dotenv
import os
from IPython.display import display, Image
import langgraph
from helper_functions import (
    text_wrap,
)
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
    config = {"recursion_limit": recursion_limit}
    try:
        # Stream the outputs from the plan_and_execute_app workflow
        for plan_output in plan_and_execute_app.stream(inputs, config=config):
            for _, agent_state_value in plan_output.items():
                pass  # agent_state_value holds the latest state after each node execution
                print(f" curr step: {agent_state_value}")
        response = agent_state_value["response"]
    except langgraph.errors.GraphRecursionError:
        response = "The answer wasn't found in the data."
    # Save the final state for further inspection or evaluation
    final_state = agent_state_value
    print(text_wrap(f" the final answer is: {response}"))
    return response, final_state


#
if __name__ == "__main__":
    initial_input = {
        "question": "who is Harry Potter?",
        "past_steps": [],
        "aggregated_context": "",
        "tool": "",
    }
    final_answer, final_state = execute_plan_and_print_steps(initial_input)
    print("=" * 50)
    print(final_answer)