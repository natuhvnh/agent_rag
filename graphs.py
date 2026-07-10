from langgraph.graph import END, StateGraph
from langchain_core.runnables import RunnableConfig
from pprint import pprint
from models import (
    QualitativeRetrievalGraphState,
    PlanExecute,
    QualitativeAnswerGraphState,
)
from nodes import (
    retrieve_chunks_context_per_question,
    keep_only_relevant_content,
    is_distilled_content_grounded_on_content,
    retrieve_summaries_context_per_question,
    retrieve_book_quotes_context_per_question,
    answer_question_from_context,
    is_answer_grounded_on_context,
    anonymize_queries,
    plan_step,
    deanonymize_queries,
    run_task_handler_chain,
    break_down_plan_step,
    replan_step,
    retrieve_or_answer,
    can_be_answered,
    rewrite_queries
)


#
def build_chunks_retrieval_workflow():
    workflow = StateGraph(QualitativeRetrievalGraphState)
    workflow.add_node(
        "retrieve_chunks_context_per_question", retrieve_chunks_context_per_question
    )
    workflow.add_node("keep_only_relevant_content", keep_only_relevant_content)
    workflow.set_entry_point("retrieve_chunks_context_per_question")
    workflow.add_edge(
        "retrieve_chunks_context_per_question", "keep_only_relevant_content"
    )
    workflow.add_conditional_edges(
        "keep_only_relevant_content",
        is_distilled_content_grounded_on_content,
        {
            "grounded on the original context": END,
            "not grounded on the original context": "keep_only_relevant_content",
        },
    )
    return workflow.compile()


#
def build_qualitative_summaries_retrieval_workflow():
    workflow = StateGraph(QualitativeRetrievalGraphState)
    workflow.add_node(
        "retrieve_summaries_context_per_question",
        retrieve_summaries_context_per_question,
    )
    workflow.add_node("keep_only_relevant_content", keep_only_relevant_content)
    workflow.set_entry_point("retrieve_summaries_context_per_question")
    workflow.add_edge(
        "retrieve_summaries_context_per_question", "keep_only_relevant_content"
    )
    workflow.add_conditional_edges(
        "keep_only_relevant_content",
        is_distilled_content_grounded_on_content,
        {
            "grounded on the original context": END,
            "not grounded on the original context": "keep_only_relevant_content",
        },
    )
    return workflow.compile()


#
def build_qualitative_book_quotes_retrieval_workflow():
    workflow = StateGraph(QualitativeRetrievalGraphState)
    workflow.add_node(
        "retrieve_book_quotes_context_per_question",
        retrieve_book_quotes_context_per_question,
    )
    workflow.add_node("keep_only_relevant_content", keep_only_relevant_content)
    workflow.set_entry_point("retrieve_book_quotes_context_per_question")
    workflow.add_edge(
        "retrieve_book_quotes_context_per_question", "keep_only_relevant_content"
    )
    workflow.add_conditional_edges(
        "keep_only_relevant_content",
        is_distilled_content_grounded_on_content,
        {
            "grounded on the original context": END,
            "not grounded on the original context": "keep_only_relevant_content",
        },
    )
    return workflow.compile()


#
def build_qualitative_answer_workflow():
    workflow = StateGraph(QualitativeAnswerGraphState)
    workflow.add_node("answer_question_from_context", answer_question_from_context)
    workflow.set_entry_point("answer_question_from_context")
    workflow.add_conditional_edges(
        "answer_question_from_context",
        is_answer_grounded_on_context,
        {"hallucination": "answer_question_from_context", "grounded on context": END},
    )
    return workflow.compile()


#
def build_agent_workflow():
    agent_workflow = StateGraph(PlanExecute)
    agent_workflow.add_node("rewrite_question", rewrite_queries)
    agent_workflow.add_node("anonymize_question", anonymize_queries)
    agent_workflow.add_node("planner", plan_step)
    agent_workflow.add_node("de_anonymize_plan", deanonymize_queries)
    agent_workflow.add_node("break_down_plan", break_down_plan_step)
    agent_workflow.add_node("task_handler", run_task_handler_chain)
    agent_workflow.add_node(
        "retrieve_chunks", run_qualitative_chunks_retrieval_workflow
    )
    agent_workflow.add_node(
        "retrieve_summaries", run_qualitative_summaries_retrieval_workflow
    )
    agent_workflow.add_node(
        "retrieve_book_quotes", run_qualitative_book_quotes_retrieval_workflow
    )
    agent_workflow.add_node("answer", run_qualitative_answer_workflow)
    agent_workflow.add_node("replan", replan_step)
    agent_workflow.add_node(
        "get_final_answer", run_qualitative_answer_workflow_for_final_answer
    )
    agent_workflow.set_entry_point("rewrite_question")
    agent_workflow.add_edge("rewrite_question", "anonymize_question")
    agent_workflow.add_edge("anonymize_question", "planner")
    agent_workflow.add_edge("planner", "de_anonymize_plan")
    agent_workflow.add_edge("de_anonymize_plan", "break_down_plan")
    agent_workflow.add_edge("break_down_plan", "task_handler")
    agent_workflow.add_conditional_edges(
        "task_handler",
        retrieve_or_answer,
        {
            "chosen_tool_is_retrieve_chunks": "retrieve_chunks",
            "chosen_tool_is_retrieve_summaries": "retrieve_summaries",
            "chosen_tool_is_retrieve_quotes": "retrieve_book_quotes",
            "chosen_tool_is_answer": "answer",
        },
    )
    agent_workflow.add_edge("retrieve_chunks", "replan")
    agent_workflow.add_edge("retrieve_summaries", "replan")
    agent_workflow.add_edge("retrieve_book_quotes", "replan")
    agent_workflow.add_edge("answer", "replan")
    agent_workflow.add_conditional_edges(
        "replan",
        can_be_answered,
        {
            "can_be_answered_already": "get_final_answer",
            "cannot_be_answered_yet": "break_down_plan",
        },
    )
    agent_workflow.add_edge("get_final_answer", END)
    final_workflow = agent_workflow.compile()
    #
    graph_png = final_workflow.get_graph(xray=True).draw_mermaid_png()
    with open("rag_agent_workflow.png", "wb") as f:
        f.write(graph_png)
    return final_workflow


def run_qualitative_chunks_retrieval_workflow(state, config: RunnableConfig):
    """
    Run the qualitative chunks retrieval workflow.

    Args:
        state: The current state of the plan execution.
        config: The run's config; only its recursion_limit is reused for this sub-graph.

    Returns:
        The state with the updated aggregated context.
    """
    state["curr_state"] = "retrieve_chunks"
    print("Running the qualitative chunks retrieval workflow...")
    question = state["query_to_retrieve_or_answer"]
    inputs = {"question": question}
    sub_config = {"recursion_limit": config.get("recursion_limit")}
    # Stream outputs from the workflow app
    for output in qualitative_chunks_retrieval_workflow_app.stream(inputs, config=sub_config):
        for _, value in output.items():
            pass
        pprint("--------------------")
    # Aggregate the retrieved context
    if not state.get("aggregated_context"):
        state["aggregated_context"] = ""
    state["aggregated_context"] += value["relevant_context"]
    return state


def run_qualitative_summaries_retrieval_workflow(state, config: RunnableConfig):
    """
    Run the qualitative summaries retrieval workflow.

    Args:
        state: The current state of the plan execution.
        config: The run's config; only its recursion_limit is reused for this sub-graph.

    Returns:
        The state with the updated aggregated context.
    """
    state["curr_state"] = "retrieve_summaries"
    print("Running the qualitative summaries retrieval workflow...")
    question = state["query_to_retrieve_or_answer"]
    inputs = {"question": question}
    sub_config = {"recursion_limit": config.get("recursion_limit")}
    for output in qualitative_summaries_retrieval_workflow_app.stream(inputs, config=sub_config):
        for _, value in output.items():
            pass
        pprint("--------------------")
    if not state.get("aggregated_context"):
        state["aggregated_context"] = ""
    state["aggregated_context"] += value["relevant_context"]
    return state


def run_qualitative_book_quotes_retrieval_workflow(state, config: RunnableConfig):
    """
    Run the qualitative book quotes retrieval workflow.

    Args:
        state: The current state of the plan execution.
        config: The run's config; only its recursion_limit is reused for this sub-graph.

    Returns:
        The state with the updated aggregated context.
    """
    state["curr_state"] = "retrieve_book_quotes"
    print("Running the qualitative book quotes retrieval workflow...")
    question = state["query_to_retrieve_or_answer"]
    inputs = {"question": question}
    sub_config = {"recursion_limit": config.get("recursion_limit")}
    for output in qualitative_book_quotes_retrieval_workflow_app.stream(inputs, config=sub_config):
        for _, value in output.items():
            pass
        pprint("--------------------")
    if not state.get("aggregated_context"):
        state["aggregated_context"] = ""
    state["aggregated_context"] += value["relevant_context"]
    return state


def run_qualitative_answer_workflow(state, config: RunnableConfig):
    """
    Run the qualitative answer workflow.

    Args:
        state: The current state of the plan execution.
        config: The run's config; only its recursion_limit is reused for this sub-graph.

    Returns:
        The state with the updated aggregated context.
    """
    state["curr_state"] = "answer"
    print("Running the qualitative answer workflow...")
    question = state["query_to_retrieve_or_answer"]
    context = state["curr_context"]
    inputs = {"question": question, "context": context}
    sub_config = {"recursion_limit": config.get("recursion_limit")}
    for output in qualitative_answer_workflow_app.stream(inputs, config=sub_config):
        for _, value in output.items():
            pass
        pprint("--------------------")
    if not state.get("aggregated_context"):
        state["aggregated_context"] = ""
    state["aggregated_context"] += value["answer"]
    return state


def run_qualitative_answer_workflow_for_final_answer(state, config: RunnableConfig):
    """
    Run the qualitative answer workflow for the final answer.

    Args:
        state: The current state of the plan execution.
        config: The run's config; only its recursion_limit is reused for this sub-graph.

    Returns:
        The state with the updated response.
    """
    state["curr_state"] = "get_final_answer"
    print("Running the qualitative answer workflow for final answer...")
    question = state["question"]
    context = state["aggregated_context"]
    inputs = {"question": question, "context": context}
    sub_config = {"recursion_limit": config.get("recursion_limit")}
    for output in qualitative_answer_workflow_app.stream(inputs, config=sub_config):
        for _, value in output.items():
            pass
        pprint("--------------------")
    state["response"] = value["answer"]
    return state


# Expose compiled apps
qualitative_chunks_retrieval_workflow_app = build_chunks_retrieval_workflow()
qualitative_summaries_retrieval_workflow_app = (
    build_qualitative_summaries_retrieval_workflow()
)
qualitative_book_quotes_retrieval_workflow_app = (
    build_qualitative_book_quotes_retrieval_workflow()
)
qualitative_answer_workflow_app = build_qualitative_answer_workflow()
plan_and_execute_app = build_agent_workflow()
