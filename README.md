# Controllable RAG Agent

A "Controllable RAG" agent built with LangChain + LangGraph. It uses a plan-and-execute loop that
anonymizes the question, plans retrieval/answer steps, and self-corrects when retrieved context
or generated answers aren't grounded in the source material.

## Running

```bash
python3 rag_agent.py          # run the agent on a hardcoded sample question
jupyter lab data_processing.ipynb   # (re)build chapter summaries / chunks / quotes from the PDF
```

Requires an `.env` file configuring an Azure-hosted OpenAI-compatible endpoint (chat model +
embeddings).

## How it works

1. **Anonymize** the question (named entities → `X`/`Y`/`Z`) so the planner isn't biased by prior
   knowledge of the entities involved.
2. **Plan** an ordered list of steps to answer it, then de-anonymize and break each step down into
   a task for exactly one of four tools.
3. **Retrieve or answer**, per step, from one of three FAISS vector stores over the book (raw text
   chunks, chapter summaries, or extracted quotes) or answer directly from context gathered so far.
   Each retrieval loop self-corrects if the distilled content isn't grounded in what was retrieved.
4. **Replan** after each step: check whether the aggregated context now answers the original
   question; if not, keep going, guarded by a recursion limit.
5. **Answer** the original question from the aggregated context, with a self-correcting loop that
   rejects hallucinated (ungrounded) answers.