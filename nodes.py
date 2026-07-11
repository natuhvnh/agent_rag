from pprint import pprint
from models import PlanExecute
from collections import defaultdict
from dependencies import (
    keep_only_relevant_content_chain,
    question_answer_from_context_cot_chain,
    is_distilled_content_grounded_on_content_chain,
    is_grounded_on_facts_chain,
    task_handler_chain,
    chapter_summaries_query_retriever,
    book_quotes_query_retriever,
    chunks_query_retriever,
    bm25_retriever,
    break_down_plan_chain,
    anonymize_question_chain,
    planner,
    replanner,
    can_be_answered_already_chain,
    de_anonymize_plan_chain,
    rewrite_question_chain,
)
from helper_functions import text_wrap

# Cap on how many times the distill/ground loop may run before falling back to the raw
# retrieved context, so the loop can never spin indefinitely on a persistent "not grounded" verdict.
MAX_DISTILL_ATTEMPTS = 1

# Cap on how many times the answer/ground loop may run before falling back to the last
# generated (best-effort) answer, so it can never spin indefinitely on a persistent hallucination verdict.
MAX_ANSWER_ATTEMPTS = 1


#
def keep_only_relevant_content(state):
    """
    Filters and keeps only the relevant content from the retrieved documents that is relevant to the query.
    Args:
        state (dict): A dictionary containing:
            - "question": The query question.
            - "context": The retrieved documents as a string.
            - "distill_attempts": Number of distill attempts made so far (defaults to 0).
            - "grounding_feedback": Explanation from the previous grounding check, if any.
    Returns:
        dict: A dictionary containing:
            - "relevant_context": The filtered relevant content.
            - "context": The original context.
            - "question": The original question.
            - "distill_attempts": The incremented attempt count.
    """
    question = state["question"]
    context = state["context"]
    attempts = state.get("distill_attempts", 0)
    grounding_feedback = state.get("grounding_feedback", "")
    feedback = (
        f"Your previous attempt was rejected because it was not grounded on the retrieved "
        f"documents. Reason: {grounding_feedback} Only keep text present in the retrieved "
        f"documents; do not add or paraphrase."
        if grounding_feedback
        else ""
    )
    input_data = {
        "query": question,
        "retrieved_documents": context,
        "feedback": feedback,
    }
    print("Keeping only the relevant content...")
    pprint("--------------------")
    output = keep_only_relevant_content_chain.invoke(input_data)
    relevant_content = output.relevant_content
    relevant_content = "".join(relevant_content)

    return {
        "relevant_context": relevant_content,
        "context": context,
        "question": question,
        "distill_attempts": attempts + 1,
    }


def answer_question_from_context(state):
    """
    Answers a question from a given context using chain-of-thought reasoning.
    Args:
        state (dict): A dictionary containing:
            - "question": The query question.
            - "context" or "aggregated_context": The context to answer the question from.
            - "answer_attempts": Number of answer attempts made so far (defaults to 0).
            - "grounding_feedback": Explanation from the previous grounding check, if any.
    Returns:
        dict: A dictionary containing:
            - "answer": The answer to the question from the context.
            - "context": The context used.
            - "question": The original question.
            - "answer_attempts": The incremented attempt count.
    """
    # Use 'aggregated_context' if available, otherwise fall back to 'context'
    question = state["question"]
    context = (
        state["aggregated_context"]
        if "aggregated_context" in state
        else state["context"]
    )
    attempts = state.get("answer_attempts", 0)
    grounding_feedback = state.get("grounding_feedback", "")
    feedback = (
        f"Your previous answer was rejected because it was not grounded in the context. "
        f"Reason: {grounding_feedback} Base your answer strictly on the given context; do "
        f"not use outside knowledge."
        if grounding_feedback
        else ""
    )

    input_data = {"question": question, "context": context, "feedback": feedback}
    print("Answering the question from the retrieved context...")
    # Invoke the LLM chain to get the answer
    output = question_answer_from_context_cot_chain.invoke(input_data)
    answer = output.answer_based_on_content
    print(f"answer before checking hallucination: {answer}")
    return {
        "answer": answer,
        "context": context,
        "question": question,
        "answer_attempts": attempts + 1,
    }


def check_distilled_content_grounded(state):
    """
    Checks if the distilled content is grounded on the original context, and records the
    outcome (plus feedback for the next distill attempt) in state so the loop can
    self-correct and is guaranteed to terminate within MAX_DISTILL_ATTEMPTS.

    Args:
        state (dict): A dictionary containing:
            - "relevant_context": The distilled content.
            - "context": The original context.
            - "distill_attempts": Number of distill attempts made so far.

    Returns:
        dict: A dictionary containing:
            - "grounded": Whether the distilled content is grounded on the original context.
            - "grounding_feedback": Explanation to feed back into the next distill attempt
              (empty once grounded or once attempts are exhausted).
            - "relevant_context": The distilled content, or the raw original context if
              attempts are exhausted without a grounded result.
    """
    pprint("--------------------")
    print("Determining if the distilled content is grounded on the original context...")
    distilled_content = state["relevant_context"]
    original_context = state["context"]
    attempts = state.get("distill_attempts", 0)

    input_data = {
        "distilled_content": distilled_content,
        "original_context": original_context,
    }

    # Invoke the LLM chain to check grounding
    output = is_distilled_content_grounded_on_content_chain.invoke(input_data)
    grounded = output["grounded"]

    if grounded:
        print("The distilled content is grounded on the original context.")
        return {
            "grounded": True,
            "grounding_feedback": "",
            "relevant_context": distilled_content,
        }

    print("The distilled content is not grounded on the original context.")
    if attempts >= MAX_DISTILL_ATTEMPTS:
        print(
            f"Giving up after {attempts} distill attempts; falling back to the raw retrieved context."
        )
        return {
            "grounded": False,
            "grounding_feedback": "",
            "relevant_context": original_context,
        }

    return {
        "grounded": False,
        "grounding_feedback": output.get("explanation", ""),
        "relevant_context": distilled_content,
    }


def route_after_grounding(state):
    """
    Routes the retrieval sub-graph based on the grounding check outcome.

    Args:
        state (dict): A dictionary containing:
            - "grounded": Whether the distilled content is grounded on the original context.
            - "distill_attempts": Number of distill attempts made so far.

    Returns:
        str: "grounded on the original context" if grounded or attempts are exhausted
             (ending the loop), otherwise "not grounded on the original context" (loops
             back to re-distill with feedback).
    """
    if state["grounded"] or state.get("distill_attempts", 0) >= MAX_DISTILL_ATTEMPTS:
        return "grounded on the original context"
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
    top_k = 5
    rrf_k = 60
    print("Retrieving relevant chunks...")
    question = state["question"]

    bm25_docs = bm25_retriever.invoke(question)
    vector_docs = chunks_query_retriever.invoke(question)

    scores = defaultdict(float)
    unique_docs = {}

    # BM25 ranking
    for rank, doc in enumerate(bm25_docs):
        key = doc.page_content

        unique_docs[key] = doc
        scores[key] += 1.0 / (rrf_k + rank + 1)

    # Vector ranking
    for rank, doc in enumerate(vector_docs):
        key = doc.page_content

        unique_docs[key] = doc
        scores[key] += 1.0 / (rrf_k + rank + 1)

    # Sort by RRF score
    ranked_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    # Return final documents
    top_docs = [unique_docs[key] for key, _ in ranked_scores[:top_k]]
    # Concatenate the content of the retrieved documents
    context = " ".join(doc.page_content for doc in top_docs)
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
    return {"context": book_quotes, "question": question}


def check_answer_grounded_on_context(state):
    """
    Checks if the answer to the question is grounded in the provided context (i.e., not a
    hallucination), and records the outcome (plus feedback for the next answer attempt) in
    state so the loop can self-correct and is guaranteed to terminate within
    MAX_ANSWER_ATTEMPTS.

    Args:
        state (dict): A dictionary containing:
            - "context": The context used to answer the question.
            - "answer": The generated answer to the question.
            - "answer_attempts": Number of answer attempts made so far.

    Returns:
        dict: A dictionary containing:
            - "grounded": Whether the answer is grounded in the context.
            - "grounding_feedback": Explanation to feed back into the next answer attempt
              (empty once grounded or once attempts are exhausted).
            - "answer": The generated answer (best-effort, unchanged, even if attempts are
              exhausted without a grounded result).
    """
    print("Checking if the answer is grounded in the facts...")

    # Extract context and answer from the state
    context = state["context"]
    answer = state["answer"]
    attempts = state.get("answer_attempts", 0)

    # Use the LLM chain to check if the answer is grounded in the context
    result = is_grounded_on_facts_chain.invoke({"context": context, "answer": answer})
    grounded_on_facts = result.grounded_on_facts

    if grounded_on_facts:
        print("The answer is grounded in the facts.")
        return {"grounded": True, "grounding_feedback": "", "answer": answer}

    print("The answer is hallucination.")
    if attempts >= MAX_ANSWER_ATTEMPTS:
        print(
            f"Giving up after {attempts} answer attempts; forwarding the last (unverified) answer."
        )
        return {"grounded": False, "grounding_feedback": "", "answer": answer}

    return {
        "grounded": False,
        "grounding_feedback": result.explanation,
        "answer": answer,
    }


def route_after_answer_grounding(state):
    """
    Routes the answer sub-graph based on the grounding check outcome.

    Args:
        state (dict): A dictionary containing:
            - "grounded": Whether the answer is grounded in the context.
            - "answer_attempts": Number of answer attempts made so far.

    Returns:
        str: "grounded on context" if grounded or attempts are exhausted (ending the loop),
             otherwise "hallucination" (loops back to re-answer with feedback).
    """
    if state["grounded"] or state.get("answer_attempts", 0) >= MAX_ANSWER_ATTEMPTS:
        return "grounded on context"
    return "hallucination"


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


def rewrite_queries(state: PlanExecute):
    """
    Re-write the question.

    Args:
        state: The current state of the plan execution.

    Returns:
        The updated state with the rewritten questions for semantic and keyword search.
    """
    state["curr_state"] = "rewrite_question"
    print("Re-write question")
    pprint("--------------------")
    rewrite_question_output = rewrite_question_chain.invoke(state["question"])
    rewritten_question = rewrite_question_output["rewritten_question"]
    keyword_question = rewrite_question_output["keyword_question"]
    print(f"rewrite_querry: {rewritten_question} AND {keyword_question}")
    pprint("--------------------")
    state["original_question"] = state["question"]
    state["question"] = rewritten_question
    state["keyword_question"] = keyword_question
    return state


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
