import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from models import (
    KeepRelevantContent,
    IsDistilledContentGroundedOnContent,
    IsGroundedOnFacts,
    QuestionAnswer,
    QuestionAnswerFromContext,
    TaskHandlerOutput,
    AnonymizeQuestion,
    DeAnonymizePlan,
    CanBeAnsweredAlready,
    Plan,
    ActPossibleResults,
)
from prompts import (
    is_grounded_on_facts_prompt_template,
    keep_only_relevant_content_prompt_template,
    can_be_answered_prompt_template,
    is_distilled_content_grounded_on_content_prompt_template,
    question_answer_cot_prompt_template,
    tasks_handler_prompt_template,
    anonymize_question_prompt_template,
    can_be_answered_already_prompt_template,
    de_anonymize_plan_prompt_template,
    break_down_plan_prompt_template,
    planner_prompt_template,
    replanner_prompt_template,
)

load_dotenv()
# 1. Initialize LLMs and Embeddings
azure_llm_key = os.getenv("azure_llm_key")
llm = ChatOpenAI(
    model="DeepSeek-V4-Flash",
    base_url="https://3t-ai-resource.services.ai.azure.com/openai/v1",
    api_key=azure_llm_key,
    max_tokens=2048,
    temperature=0.1,
)
#
embedding_base_url = os.getenv("embedding_base_url")
embedding_key = os.getenv("embedding_key")
embedding_deployment = os.getenv("embedding_deployment")
embeddings = OpenAIEmbeddings(
    model=embedding_deployment,
    base_url=f"{embedding_base_url}/openai/v1",
    api_key=embedding_key,
)
# 2. Initialize Retrievers
chunks_vector_store = FAISS.load_local(
    "embedding/chunks_vector_store",
    embeddings,
    allow_dangerous_deserialization=True,
)
chapter_summaries_vector_store = FAISS.load_local(
    "embedding/chapter_summaries_vector_store",
    embeddings,
    allow_dangerous_deserialization=True,
)
book_quotes_vectorstore = FAISS.load_local(
    "embedding/book_quotes_vectorstore",
    embeddings,
    allow_dangerous_deserialization=True,
)
chunks_query_retriever = chunks_vector_store.as_retriever(search_kwargs={"k": 5})
chapter_summaries_query_retriever = chapter_summaries_vector_store.as_retriever(
    search_kwargs={"k": 2}
)
book_quotes_query_retriever = book_quotes_vectorstore.as_retriever(
    search_kwargs={"k": 5}
)
# 3. Initialize Chains
# Create the LLM chain for fact-checking
is_grounded_on_facts_prompt = PromptTemplate(
    template=is_grounded_on_facts_prompt_template,
    input_variables=["context", "answer"],
)
is_grounded_on_facts_chain = is_grounded_on_facts_prompt | llm.with_structured_output(
    IsGroundedOnFacts
)
# Create the prompt object for the LLM
can_be_answered_json_parser = JsonOutputParser(pydantic_object=QuestionAnswer)
answer_question_prompt = PromptTemplate(
    template=can_be_answered_prompt_template,
    input_variables=["question", "context"],
    partial_variables={
        "format_instructions": can_be_answered_json_parser.get_format_instructions()
    },
)
can_be_answered_chain = answer_question_prompt | llm | can_be_answered_json_parser
#
# Create the chain for filtering retrieved content down to what's relevant to the query
keep_only_relevant_content_prompt = PromptTemplate(
    template=keep_only_relevant_content_prompt_template,
    input_variables=["query", "retrieved_documents"],
)
keep_only_relevant_content_chain = (
    keep_only_relevant_content_prompt | llm.with_structured_output(KeepRelevantContent)
)
# Create the chain for answering a question from context using chain-of-thought reasoning
question_answer_from_context_cot_prompt = PromptTemplate(
    template=question_answer_cot_prompt_template,
    input_variables=["context", "question"],
)
question_answer_from_context_cot_chain = (
    question_answer_from_context_cot_prompt
    | llm.with_structured_output(QuestionAnswerFromContext)
)
# Create the chain for checking if distilled content is grounded on the original context
is_distilled_content_grounded_on_content_json_parser = JsonOutputParser(
    pydantic_object=IsDistilledContentGroundedOnContent
)
is_distilled_content_grounded_on_content_prompt = PromptTemplate(
    template=is_distilled_content_grounded_on_content_prompt_template,
    input_variables=["distilled_content", "original_context"],
    partial_variables={
        "format_instructions": is_distilled_content_grounded_on_content_json_parser.get_format_instructions()
    },
)
is_distilled_content_grounded_on_content_chain = (
    is_distilled_content_grounded_on_content_prompt
    | llm
    | is_distilled_content_grounded_on_content_json_parser
)
#
task_handler_prompt = PromptTemplate(
    template=tasks_handler_prompt_template,
    input_variables=[
        "curr_task",
        "aggregated_context",
        "last_tool",
        "past_steps",
        "question",
    ],
)
task_handler_chain = task_handler_prompt | llm.with_structured_output(TaskHandlerOutput)
#
anonymize_question_parser = JsonOutputParser(pydantic_object=AnonymizeQuestion)
anonymize_question_prompt = PromptTemplate(
    template=anonymize_question_prompt_template,
    input_variables=["question"],
    partial_variables={
        "format_instructions": anonymize_question_parser.get_format_instructions()
    },
)
anonymize_question_chain = anonymize_question_prompt | llm | anonymize_question_parser
#
de_anonymize_plan_prompt = PromptTemplate(
    template=de_anonymize_plan_prompt_template,
    input_variables=["plan", "mapping"],
)
de_anonymize_plan_chain = de_anonymize_plan_prompt | llm.with_structured_output(
    DeAnonymizePlan
)
#
can_be_answered_already_prompt = PromptTemplate(
    template=can_be_answered_already_prompt_template,
    input_variables=["question", "context"],
)
can_be_answered_already_chain = (
    can_be_answered_already_prompt | llm.with_structured_output(CanBeAnsweredAlready)
)
#
break_down_plan_prompt = PromptTemplate(
    template=break_down_plan_prompt_template,
    input_variables=["plan"],
)
break_down_plan_chain = break_down_plan_prompt | llm.with_structured_output(Plan)
#
planner_prompt = PromptTemplate(
    template=planner_prompt_template,
    input_variables=["question"],
)
planner = planner_prompt | llm.with_structured_output(Plan)
#
act_possible_results_parser = JsonOutputParser(pydantic_object=ActPossibleResults)
replanner_prompt = PromptTemplate(
    template=replanner_prompt_template,
    input_variables=["question", "plan", "past_steps", "aggregated_context"],
    partial_variables={
        "format_instructions": act_possible_results_parser.get_format_instructions()
    },
)
replanner = replanner_prompt | llm | act_possible_results_parser
