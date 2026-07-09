from pprint import pprint
from models import PlanExecute
from dependencies import (
    keep_only_relevant_content_chain,
    question_answer_from_context_cot_chain,
    is_distilled_content_grounded_on_content_chain,
    is_grounded_on_facts_chain,
    task_handler_chain,
    chapter_summaries_query_retriever,
    book_quotes_query_retriever,
    chunks_query_retriever,
    break_down_plan_chain,
    anonymize_question_chain,
    planner,
    replanner,
    can_be_answered_already_chain,
    de_anonymize_plan_chain,
)
from helper_functions import escape_quotes, text_wrap


#
def keep_only_relevant_content(state):
    """
    Filters and keeps only the relevant content from the retrieved documents that is relevant to the query.
    Args:
        state (dict): A dictionary containing:
            - "question": The query question.
            - "context": The retrieved documents as a string.
    Returns:
        dict: A dictionary containing:
            - "relevant_context": The filtered relevant content.
            - "context": The original context.
            - "question": The original question.
    """
    question = state["question"]
    context = state["context"]
    input_data = {"query": question, "retrieved_documents": context}
    print("Keeping only the relevant content...")
    pprint("--------------------")
    output = keep_only_relevant_content_chain.invoke(input_data)
    relevant_content = output.relevant_content
    relevant_content = "".join(relevant_content)
    relevant_content = escape_quotes(relevant_content)

    return {
        "relevant_context": relevant_content,
        "context": context,
        "question": question,
    }


def answer_question_from_context(state):
    """
    Answers a question from a given context using chain-of-thought reasoning.
    Args:
        state (dict): A dictionary containing:
            - "question": The query question.
            - "context" or "aggregated_context": The context to answer the question from.
    Returns:
        dict: A dictionary containing:
            - "answer": The answer to the question from the context.
            - "context": The context used.
            - "question": The original question.
    """
    # Use 'aggregated_context' if available, otherwise fall back to 'context'
    question = state["question"]
    context = (
        state["aggregated_context"]
        if "aggregated_context" in state
        else state["context"]
    )

    input_data = {"question": question, "context": context}
    print("Answering the question from the retrieved context...")
    # Invoke the LLM chain to get the answer
    output = question_answer_from_context_cot_chain.invoke(input_data)
    answer = output.answer_based_on_content
    print(f"answer before checking hallucination: {answer}")
    return {"answer": answer, "context": context, "question": question}


def is_distilled_content_grounded_on_content(state):
    """
    Determines if the distilled content is grounded on the original context.

    Args:
        state (dict): A dictionary containing:
            - "relevant_context": The distilled content.
            - "context": The original context.

    Returns:
        str: "grounded on the original context" if grounded, otherwise "not grounded on the original context".
    """
    pprint("--------------------")
    print("Determining if the distilled content is grounded on the original context...")
    distilled_content = state["relevant_context"]
    original_context = state["context"]

    input_data = {
        "distilled_content": distilled_content,
        "original_context": original_context,
    }

    # Invoke the LLM chain to check grounding
    output = is_distilled_content_grounded_on_content_chain.invoke(input_data)
    grounded = output["grounded"]

    if grounded:
        print("The distilled content is grounded on the original context.")
        return "grounded on the original context"
    else:
        print("The distilled content is not grounded on the original context.")
        return "not grounded on the original context"


def retrieve_chunks_context_per_question(state):
    """
    Retrieves relevant context for a given question from the book chunks.

    Args:
        state (dict): A dictionary containing the question to answer, with key "question".

    Returns:
        dict: A dictionary with keys:
            - "context": Aggregated context string from relevant book chunks.
            - "question": The original question.
    """
    print("Retrieving relevant chunks...")
    question = state["question"]
    # Retrieve relevant book chunks using the retriever
    docs = chunks_query_retriever.invoke(question)
    # Concatenate the content of the retrieved documents
    context = " ".join(doc.page_content for doc in docs)
    context = escape_quotes(context)
    return {"context": context, "question": question}


def retrieve_summaries_context_per_question(state):
    """
    Retrieves relevant context for a given question from chapter summaries.

    Args:
        state (dict): A dictionary containing the question to answer, with key "question".

    Returns:
        dict: A dictionary with keys:
            - "context": Aggregated context string from relevant chapter summaries.
            - "question": The original question.
    """
    print("Retrieving relevant chapter summaries...")
    question = state["question"]
    # Retrieve relevant chapter summaries using the retriever
    docs_summaries = chapter_summaries_query_retriever.invoke(question)
    # Concatenate the content of the retrieved summaries, including chapter citation
    context_summaries = " ".join(
        f"{doc.page_content} (Chapter {doc.metadata['chapter']})"
        for doc in docs_summaries
    )
    context_summaries = escape_quotes(context_summaries)
    return {"context": context_summaries, "question": question}


def retrieve_book_quotes_context_per_question(state):
    """
    Retrieves relevant context for a given question from book quotes.

    Args:
        state (dict): A dictionary containing the question to answer, with key "question".

    Returns:
        dict: A dictionary with keys:
            - "context": Aggregated context string from relevant book quotes.
            - "question": The original question.
    """
    question = state["question"]
    print("Retrieving relevant book quotes...")
    # Retrieve relevant book quotes using the retriever
    docs_book_quotes = book_quotes_query_retriever.invoke(question)
    # Concatenate the content of the retrieved quotes
    book_quotes = " ".join(doc.page_content for doc in docs_book_quotes)
    book_quotes_context = escape_quotes(book_quotes)
    return {"context": book_quotes_context, "question": question}


def is_answer_grounded_on_context(state):
    """
    Determines if the answer to the question is grounded in the provided context (i.e., not a hallucination).

    Args:
        state (dict): A dictionary containing:
            - "context": The context used to answer the question.
            - "answer": The generated answer to the question.

    Returns:
        str: "hallucination" if the answer is not grounded in the context,
             "grounded on context" if the answer is supported by the context.
    """
    print("Checking if the answer is grounded in the facts...")

    # Extract context and answer from the state
    context = state["context"]
    answer = state["answer"]

    # Use the LLM chain to check if the answer is grounded in the context
    result = is_grounded_on_facts_chain.invoke({"context": context, "answer": answer})
    grounded_on_facts = result.grounded_on_facts

    # Return the result based on grounding
    if not grounded_on_facts:
        print("The answer is hallucination.")
        return "hallucination"
    else:
        print("The answer is grounded in the facts.")
        return "grounded on context"


def run_task_handler_chain(state: PlanExecute):
    """
    Run the task handler chain to decide which tool to use to execute the task.

    Args:
        state: The current state of the plan execution.

    Returns:
        The updated state of the plan execution.
    """
    state["curr_state"] = "task_handler"
    print("the current plan is:")
    print(state["plan"])
    pprint("--------------------")

    # Initialize past_steps if not present
    if not state.get("past_steps"):
        state["past_steps"] = []
    # Get the current task from the plan
    if not state["plan"]:
        raise ValueError(
            "task_handler received an empty plan; the replanner should never return an "
            "empty plan while the original question is still unanswered."
        )
    curr_task = state["plan"][0]

    # Prepare inputs for the task handler chain
    inputs = {
        "curr_task": curr_task,
        "aggregated_context": state.get("aggregated_context", ""),
        "last_tool": state.get("tool", ""),
        "past_steps": state["past_steps"],
        "question": state["question"],
    }

    # Invoke the task handler chain
    output = task_handler_chain.invoke(inputs)

    # Update state with the completed task
    state["past_steps"].append(curr_task)
    state["plan"].pop(0)

    # Decide which tool to use based on output
    if output.tool == "retrieve_chunks":
        state["query_to_retrieve_or_answer"] = output.query
        state["tool"] = "retrieve_chunks"
    elif output.tool == "retrieve_summaries":
        state["query_to_retrieve_or_answer"] = output.query
        state["tool"] = "retrieve_summaries"
    elif output.tool == "retrieve_quotes":
        state["query_to_retrieve_or_answer"] = output.query
        state["tool"] = "retrieve_quotes"
    elif output.tool == "answer_from_context":
        state["query_to_retrieve_or_answer"] = output.query
        state["curr_context"] = output.curr_context
        state["tool"] = "answer"
    else:
        raise ValueError(
            "Invalid tool was outputted by task handler. Must be one of "
            f"'retrieve_chunks', 'retrieve_summaries', 'retrieve_quotes', or 'answer_from_context'. "
            f"Got: {output.tool!r}"
        )
    return state


def retrieve_or_answer(state: PlanExecute):
    """
    Decide whether to retrieve or answer the question based on the current state.

    Args:
        state: The current state of the plan execution.

    Returns:
        String indicating the chosen tool.
    """
    state["curr_state"] = "decide_tool"
    print("deciding whether to retrieve or answer")
    if state["tool"] == "retrieve_chunks":
        return "chosen_tool_is_retrieve_chunks"
    elif state["tool"] == "retrieve_summaries":
        return "chosen_tool_is_retrieve_summaries"
    elif state["tool"] == "retrieve_quotes":
        return "chosen_tool_is_retrieve_quotes"
    elif state["tool"] == "answer":
        return "chosen_tool_is_answer"
    else:
        raise ValueError(
            "Invalid tool in state. Must be one of "
            f"'retrieve_chunks', 'retrieve_summaries', 'retrieve_quotes', or 'answer'. "
            f"Got: {state['tool']!r}"
        )


def anonymize_queries(state: PlanExecute):
    """
    Anonymizes the question.

    Args:
        state: The current state of the plan execution.

    Returns:
        The updated state with the anonymized question and mapping.
    """
    state["curr_state"] = "anonymize_question"
    print("Anonymizing question")
    pprint("--------------------")
    anonymized_question_output = anonymize_question_chain.invoke(state["question"])
    anonymized_question = anonymized_question_output["anonymized_question"]
    print(f"anonimized_querry: {anonymized_question}")
    pprint("--------------------")
    mapping = anonymized_question_output["mapping"]
    state["anonymized_question"] = anonymized_question
    state["mapping"] = mapping
    return state


def deanonymize_queries(state: PlanExecute):
    """
    De-anonymizes the plan.

    Args:
        state: The current state of the plan execution.

    Returns:
        The updated state with the de-anonymized plan.
    """
    state["curr_state"] = "de_anonymize_plan"
    print("De-anonymizing plan")
    pprint("--------------------")
    deanonimzed_plan = de_anonymize_plan_chain.invoke(
        {"plan": state["plan"], "mapping": state["mapping"]}
    )
    state["plan"] = deanonimzed_plan.plan
    print(f"de-anonimized_plan: {deanonimzed_plan.plan}")
    return state


def plan_step(state: PlanExecute):
    """
    Plans the next step.

    Args:
        state: The current state of the plan execution.

    Returns:
        The updated state with the plan.
    """
    state["curr_state"] = "planner"
    print("Planning step")
    pprint("--------------------")
    plan = planner.invoke({"question": state["anonymized_question"]})
    state["plan"] = plan.steps
    print(f'plan: {state["plan"]}')
    return state


def break_down_plan_step(state: PlanExecute):
    """
    Breaks down the plan steps into retrievable or answerable tasks.

    Args:
        state: The current state of the plan execution.

    Returns:
        The updated state with the refined plan.
    """
    state["curr_state"] = "break_down_plan"
    print("Breaking down plan steps into retrievable or answerable tasks")
    pprint("--------------------")
    refined_plan = break_down_plan_chain.invoke(state["plan"])
    state["plan"] = refined_plan.steps
    return state


def replan_step(state: PlanExecute):
    """
    Replans the next step.

    Args:
        state: The current state of the plan execution.

    Returns:
        The updated state with the plan.
    """
    state["curr_state"] = "replan"
    print("Replanning step")
    pprint("--------------------")
    inputs = {
        "question": state["question"],
        "plan": state["plan"],
        "past_steps": state["past_steps"],
        "aggregated_context": state["aggregated_context"],
    }
    output = replanner.invoke(inputs)
    state["plan"] = output["plan"]["steps"]
    return state


def can_be_answered(state: PlanExecute):
    """
    Determines if the question can be answered.

    Args:
        state: The current state of the plan execution.

    Returns:
        String indicating whether the original question can be answered or not.
    """
    state["curr_state"] = "can_be_answered_already"
    print("Checking if the ORIGINAL QUESTION can be answered already")
    pprint("--------------------")
    question = state["question"]
    context = state["aggregated_context"]
    inputs = {"question": question, "context": context}
    output = can_be_answered_already_chain.invoke(inputs)
    if output.can_be_answered:
        print("The ORIGINAL QUESTION can be fully answered already.")
        pprint("--------------------")
        print("the aggregated context is:")
        print(text_wrap(state["aggregated_context"]))
        print("--------------------")
        return "can_be_answered_already"
    else:
        print("The ORIGINAL QUESTION cannot be fully answered yet.")
        pprint("--------------------")
        return "cannot_be_answered_yet"
