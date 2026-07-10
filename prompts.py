keep_only_relevant_content_prompt_template = """
    You receive a query: {query} and retrieved documents: {retrieved_documents} from a vector store.
    You need to filter out all the non relevant information that doesn't supply important information regarding the {query}.
    Your goal is just to filter out the non relevant information.
    You can remove parts of sentences that are not relevant to the query or remove whole sentences that are not relevant to the query.
    DO NOT ADD ANY NEW INFORMATION THAT IS NOT IN THE RETRIEVED DOCUMENTS.
    Output the filtered relevant content.
    """

rewrite_prompt_template = """
You are a question re-writer that converts an input question to a better version optimized for vectorstore retrieval and keyword extraction.
Analyze the input question {question} and try to reason about the underlying semantic intent / meaning.
{format_instructions}
"""

question_answer_cot_prompt_template = """ 
Examples of Chain-of-Thought Reasoning

Example 1

Context: Mary is taller than Jane. Jane is shorter than Tom. Tom is the same height as David.
Question: Who is the tallest person?
Reasoning Chain:
The context tells us Mary is taller than Jane
It also says Jane is shorter than Tom
And Tom is the same height as David
So the order from tallest to shortest is: Mary, Tom/David, Jane
Therefore, Mary must be the tallest person

Example 2
Context: Harry was reading a book about magic spells. One spell allowed the caster to turn a person into an animal for a short time. Another spell could levitate objects.
 A third spell created a bright light at the end of the caster's wand.
Question: Based on the context, if Harry cast these spells, what could he do?
Reasoning Chain:
The context describes three different magic spells
The first spell allows turning a person into an animal temporarily
The second spell can levitate or float objects
The third spell creates a bright light
If Harry cast these spells, he could turn someone into an animal for a while, make objects float, and create a bright light source
So based on the context, if Harry cast these spells he could transform people, levitate things, and illuminate an area
Instructions.

Example 3 
Context: Harry Potter woke up on his birthday to find a present at the end of his bed. He excitedly opened it to reveal a Nimbus 2000 broomstick.
Question: Why did Harry receive a broomstick for his birthday?
Reasoning Chain:
The context states that Harry Potter woke up on his birthday and received a present - a Nimbus 2000 broomstick.
However, the context does not provide any information about why he received that specific present or who gave it to him.
There are no details about Harry's interests, hobbies, or the person who gifted him the broomstick.
Without any additional context about Harry's background or the gift-giver's motivations, there is no way to determine the reason he received a broomstick as a birthday present.

For the question below, provide your answer by first showing your step-by-step reasoning process, breaking down the problem into a chain of thought before arriving at the final answer,
 just like in the previous examples.
Context
{context}
Question
{question}
"""

is_relevant_content_prompt_template = """
You receive a query: {query} and a context: {context} retrieved from a vector store. 
You need to determine if the document is relevant to the query. 
{format_instructions}
"""

is_grounded_on_facts_prompt_template = """
You are a fact-checker that determines if the given answer {answer} is grounded in the given context {context}
You don't mind if it doesn't make sense, as long as it is grounded in the context.
Output a JSON containing the answer to the question, and apart from the JSON format don't output any additional text.
"""

can_be_answered_prompt_template = """
You receive a query: {question} and a context: {context}. 
You need to determine if the question can be fully answered based on the context.
{format_instructions}
"""

is_distilled_content_grounded_on_content_prompt_template = """
You receive some distilled content: {distilled_content} and the original context: {original_context}.
You need to determine if the distilled content is grounded on the original context.
If the distilled content is grounded on the original context, set the grounded field to true.
If the distilled content is not grounded on the original context, set the grounded field to false.
{format_instructions}
"""

planner_prompt_template = """
For the given query {question}, come up with a simple step by step plan of how to figure out the answer.
This plan should involve individual tasks, that if executed correctly will yield the correct answer. Do not add any superfluous steps. 
The result of the final step should be the final answer. Make sure that each step has all the information needed - do not skip steps.
"""

break_down_plan_prompt_template = """
You receive a plan {plan} which contains a series of steps to follow in order to answer a query. 
You need to go through the plan and refine it according to these rules:
1. Every step must be executable by one of the following:
    i. Retrieving relevant information from a vector store of book chunks
    ii. Retrieving relevant information from a vector store of chapter summaries
    iii. Retrieving relevant information from a vector store of book quotes
    iv. Answering a question from a given context.
2. Every step should contain all the information needed to execute it.
Output the refined plan.
"""

replanner_prompt_template = """
For the given objective, come up with a simple step by step plan of how to figure out the answer. 
This plan should involve individual tasks, that if executed correctly will yield the correct answer. Do not add any superfluous steps. 
The result of the final step should be the final answer. Make sure that each step has all the information needed - do not skip steps.
Assume that the answer was not found yet and you need to update the plan accordingly, so the plan should never be empty.
Your objective was this:
{question}
Your original plan was this:
{plan}
You have currently done the follow steps:
{past_steps}
You already have the following context:
{aggregated_context}
Update your plan accordingly. If further steps are needed, fill out the plan with only those steps.
Do not return previously done steps as part of the plan.
The format is JSON so escape quotes and new lines.
{format_instructions}
"""

tasks_handler_prompt_template = """
You are a task handler that receives a task {curr_task} and have to decide which tool to use to execute the task.
You have the following tools at your disposal:
retrieve_chunks: a tool that retrieves relevant information from a vector store of book chunks based on a given query.
- use retrieve_chunks when you think the current task should search for information in the book chunks.
retrieve_summaries: a tool that retrieves relevant information from a vector store of chapter summaries based on a given query.
- use retrieve_summaries when you think the current task should search for information in the chapter summaries.
retrieve_quotes: a tool that retrieves relevant information from a vector store of quotes from the book based on a given query.
- use retrieve_quotes when you think the current task should search for information in the book quotes.
answer_from_context: a tool that answers a question from a given context.
- use answer_from_context ONLY when you the current task can be answered by the aggregated context {aggregated_context}
You also receive the last tool used {last_tool}
if {last_tool} was retrieve_chunks, use other tools than retrieve_chunks.
You also have the past steps {past_steps} that you can use to make decisions and understand the context of the task.
You also have the initial user's question {question} that you can use to make decisions and understand the context of the task.
if you decide to use retrieve_chunks, retrieve_summaries, or retrieve_quotes, output the query to be used for the tool and also output the relevant tool.
if you decide to use answer_from_context, output the question to be used for the tool, the context, and also that the tool to be used is answer_from_context.
The tool field in your output must be set to exactly one of: retrieve_chunks, retrieve_summaries, retrieve_quotes, answer_from_context.
"""

anonymize_question_prompt_template = """
You are a question anonymizer. The input you receive is a string containing several words that
construct a question {question}. Your goal is to change all name entities in the input to variables, and remember the mapping of the original name entities to the variables.
Example 1:
  if the input is "who is harry potter?" the output should be "who is X?" and the mapping should be {{"X": "harry potter"}}
Example 2:
  if the input is "how did the bad guy played with the alex and rony?"
  the output should be "how did the X played with the Y and Z?" and the mapping should be {{"X": "bad guy", "Y": "alex", "Z": "rony"}}
You must replace all name entities in the input with variables, and remember the mapping of the original name entities to the variables.
Output the anonymized question and the mapping in a JSON format.
{format_instructions}
"""

de_anonymize_plan_prompt_template = (
    "You receive a list of tasks: {plan}, where some of the words are replaced with mapped variables. "
    "You also receive the mapping for those variables to words {mapping}. "
    "Replace all the variables in the list of tasks with the mapped words. "
    "If no variables are present, return the original list of tasks. "
    "In any case, just output the updated list of tasks in a JSON format as described here, "
    "without any additional text apart from the JSON."
)

can_be_answered_already_prompt_template = """
You receive a query: {question} and a context: {context}.
You need to determine if the question can be fully answered relying only on the given context.
The only information you have and can rely on is the context you received. 
You have no prior knowledge of the question or the context.
If you think the question can be answered based on the context, output 'true', otherwise output 'false'.
"""
