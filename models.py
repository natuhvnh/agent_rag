from pydantic import BaseModel, Field
from typing import List, Dict, Literal, TypedDict


class KeepRelevantContent(BaseModel):
    relevant_content: str = Field(
        description="The relevant content from the retrieved documents that is relevant to the query."
    )


class QuestionAnswerFromContext(BaseModel):
    answer_based_on_content: str = Field(
        description="Generates an answer to a query based on a given context."
    )


class IsGroundedOnFacts(BaseModel):
    """
    Output schema for checking if the answer is grounded in the provided context.
    """

    grounded_on_facts: bool = Field(
        description="Answer is grounded in the facts, 'yes' or 'no'"
    )
    explanation: str = Field(
        description="An explanation of why the answer is or is not grounded in the facts."
    )


class QuestionAnswer(BaseModel):
    can_be_answered: bool = Field(
        description="binary result of whether the question can be fully answered or not"
    )
    explanation: str = Field(
        description="An explanation of why the question can be fully answered or not."
    )


class IsDistilledContentGroundedOnContent(BaseModel):
    grounded: bool = Field(
        description="Whether the distilled content is grounded on the original context."
    )
    explanation: str = Field(
        description="An explanation of why the distilled content is or is not grounded on the original context."
    )


class QualitativeRetrievalGraphState(TypedDict):
    """
    Represents the state of the qualitative retrieval graph.

    Attributes:
        question (str): The input question to be answered.
        rewrite_question (str): The rewritten question used for vector (semantic) search.
        keyword_question (str): The rewritten question used for keyword (BM25) search.
        context (str): The context retrieved from the source (e.g., book chunks, summaries, or quotes).
        relevant_context (str): The distilled or filtered context that is most relevant to the question.
        distill_attempts (int): Number of times the distill/ground loop has run.
        grounded (bool): Whether the distilled content was judged grounded on the original context.
        grounding_feedback (str): Explanation from the grounding check, fed back into the next distill attempt.
    """

    question: str
    rewrite_question: str
    keyword_question: str
    context: str
    relevant_context: str
    distill_attempts: int
    grounded: bool
    grounding_feedback: str


class QualitativeAnswerGraphState(TypedDict):
    """
    Represents the state of the qualitative answer graph.

    Attributes:
        question (str): The input question to be answered.
        context (str): The context used to answer the question.
        answer (str): The generated answer to the question.
        answer_attempts (int): Number of times the answer/ground loop has run.
        grounded (bool): Whether the answer was judged grounded on the context.
        grounding_feedback (str): Explanation from the grounding check, fed back into the next answer attempt.
    """

    question: str
    context: str
    answer: str
    answer_attempts: int
    grounded: bool
    grounding_feedback: str


class PlanExecute(TypedDict):
    """
    Represents the state at each step of the plan execution pipeline.

    Attributes:
        curr_state (str): The current state or status of the execution.
        question (str): The user question. Can be original question or rewritten question
        rewrite_question (str): The rewritten question used for vector extraction
        keyword_question (str): The rewritten question used for keyword extraction
        anonymized_question (str): The anonymized version of the question (entities replaced with variables).
        query_to_retrieve_or_answer (str): The query to be used for retrieval or answering.
        plan (List[str]): The current plan as a list of steps to execute.
        past_steps (List[str]): List of steps that have already been executed.
        mapping (dict): Mapping of anonymized variables to original named entities.
        curr_context (str): The current context used for answering or retrieval.
        aggregated_context (str): The accumulated context from previous steps.
        tool (str): The tool or method used for the current step (e.g., retrieval, answer).
        response (str): The response or output generated at this step.
    """

    curr_state: str
    question: str
    rewrite_question: str
    keyword_question: str
    anonymized_question: str
    query_to_retrieve_or_answer: str
    plan: List[str]
    past_steps: List[str]
    mapping: Dict[str, str]
    curr_context: str
    aggregated_context: str
    tool: str
    response: str


class Plan(BaseModel):
    """
    Represents a step-by-step plan to answer a given question.
    Attributes:
        steps (List[str]): Ordered list of steps to follow.
    """

    steps: List[str] = Field(
        description="different steps to follow, should be in sorted order"
    )


class ActPossibleResults(BaseModel):
    """
    Represents the possible results of the replanning action.

    Attributes:
        plan (Plan): The updated plan to follow in the future.
        explanation (str): Explanation of the action taken or the reasoning behind the plan update.
    """

    plan: Plan = Field(description="Plan to follow in future.")
    explanation: str = Field(description="Explanation of the action.")


class TaskHandlerOutput(BaseModel):
    """
    Output schema for the task handler.
    - tool: The tool to be used; should be one of 'retrieve_chunks', 'retrieve_summaries', 'retrieve_quotes', or 'answer_from_context'.
    """
    tool: Literal[
        "retrieve_chunks",
        "retrieve_summaries",
        "retrieve_quotes",
        "answer_from_context",
    ] = Field(
        description="The tool to be used should be either retrieve_chunks, retrieve_summaries, retrieve_quotes, or answer_from_context."
    )


class RewriteQuestion(BaseModel):
    """
    Output schema for the rew-write question.
    Attributes:
      rewritten_question (str): The rewritten question for semantic search.
      keyword_question (str): The rewritten question for keyword search.
      explanation (str): Explanation of the re-write process.
    """

    rewritten_question: str = Field(description="The rewritten question for semantic search")
    keyword_question: str = Field(description="The rewritten question for keyword search")
    explanation: str = Field(description="Explanation of the re-write process")


class AnonymizeQuestion(BaseModel):
    """
    Output schema for the anonymized question.
    Attributes:
      anonymized_question (str): The question with named entities replaced by variables.
      mapping (dict): Mapping of variables to original named entities.
      explanation (str): Explanation of the anonymization process.
    """

    anonymized_question: str = Field(description="Anonymized question.")
    mapping: dict = Field(description="Mapping of original name entities to variables.")
    explanation: str = Field(description="Explanation of the action.")


class DeAnonymizePlan(BaseModel):
    """
    Output schema for the de-anonymized plan.
    Attributes:
        plan (List): Plan to follow in future, with all variables replaced by the mapped words.
    """

    plan: List = Field(
        description="Plan to follow in future. with all the variables replaced with the mapped words."
    )


class CanBeAnsweredAlready(BaseModel):
    """
    Output schema for checking if the question can be fully answered from the given context.
    Attributes:
        can_be_answered (bool): Whether the question can be fully answered or not based on the given context.
    """

    can_be_answered: bool = Field(
        description="Whether the question can be fully answered or not based on the given context."
    )
