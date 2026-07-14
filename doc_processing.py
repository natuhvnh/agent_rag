# --- LangChain and LLM Imports ---
from langchain_openai import ChatOpenAI

# --- Document Loading and Vector Store ---
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_openai import OpenAIEmbeddings
from langchain_classic.chains.summarize import load_summarize_chain
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_azure_cosmosdb import AzureCosmosDBNoSqlVectorSearch
from langchain_core.documents import Document
from azure.cosmos import CosmosClient, PartitionKey
from dotenv import load_dotenv
from time import monotonic
import tiktoken
import pickle
import json
import os
import re

# --- Datasets and Typing ---
from datasets import Dataset

# --- Helper Functions ---
from helper_functions import (
    num_tokens_from_string,
    replace_t_with_space,
    replace_double_lines_with_one_line,
    split_into_chapters,
    escape_quotes,
    text_wrap,
    extract_book_quotes_as_documents,
    tokenize,
)

# --- Load environment variables (e.g., API keys) ---
load_dotenv()


def get_tools():
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
    return llm, embeddings


def process_pdf_file(doc_path, process_quote, process_chapter):
    loader = PyPDFLoader(doc_path)
    documents = loader.load()
    document_cleaned = replace_t_with_space(documents)
    document_cleaned = replace_double_lines_with_one_line(documents)
    document_cleaned = escape_quotes(documents)
    #
    book_quotes_list = (
        extract_book_quotes_as_documents(document_cleaned) if process_quote else []
    )
    chapters = split_into_chapters(documents) if process_chapter else []
    return document_cleaned, book_quotes_list, chapters


def create_chapter_summary(chapter, llm):
    """
    Creates a summary of a chapter using a large language model (LLM).
    Args:
        chapter: A Document object representing the chapter to summarize.
    Returns:
        A Document object containing the summary of the chapter.
    """
    summarization_prompt_template = """Write an extensive summary of the following:
    {text}
    SUMMARY:"""
    summarization_prompt = PromptTemplate(
        template=summarization_prompt_template, input_variables=["text"]
    )
    # Extract the text content from the chapter
    chapter_txt = chapter.page_content
    max_tokens = 16000  # Maximum token limit for the model
    verbose = False  # Set to True for more detailed output
    num_tokens = num_tokens_from_string(chapter_txt, "gpt-4o")

    # Choose the summarization chain type based on token count
    if num_tokens < max_tokens:
        # For shorter chapters, use the "stuff" chain type
        chain = load_summarize_chain(
            llm, chain_type="stuff", prompt=summarization_prompt, verbose=verbose
        )
    else:
        # For longer chapters, use the "map_reduce" chain type
        chain = load_summarize_chain(
            llm,
            chain_type="map_reduce",
            map_prompt=summarization_prompt,
            combine_prompt=summarization_prompt,
            verbose=verbose,
        )
    start_time = monotonic()
    doc_chapter = Document(page_content=chapter_txt)
    summary_result = chain.invoke([doc_chapter])
    print(f"Chain type: {chain.__class__.__name__}")
    print(f"Run time: {monotonic() - start_time}")
    chapter_summary = Document(
        page_content=summary_result["output_text"], metadata=chapter.metadata
    )
    return chapter_summary


def create_vectorstore(documents, embeddings, database_name, container_name):
    # Create Cosmos client
    KEY = os.getenv("cosmos_key")
    ENDPOINT = os.getenv("cosmos_url")
    cosmos_client = CosmosClient(ENDPOINT, credential=KEY)
    # Vector policy
    vector_embedding_policy = {
        "vectorEmbeddings": [
            {
                "path": "/vector_embedding",
                "dataType": "float32",
                "distanceFunction": "cosine",
                "dimensions": 1536,  # Match your embedding model
            }
        ]
    }
    # Indexing policy
    indexing_policy = {
        "indexingMode": "consistent",
        "automatic": True,
        "includedPaths": [{"path": "/*"}],
        "excludedPaths": [{"path": '/"_etag"/?'}],
        "vectorIndexes": [
            {
                "path": "/vector_embedding",
                "type": "diskANN",
            }
        ],
    }
    # Create vector store and upload documents
    vector_search_fields = {"text_field": "text", "embedding_field": "vector_embedding"}
    cosmos_container_properties = {"partition_key": PartitionKey(path="/id")}
    vectorstore = AzureCosmosDBNoSqlVectorSearch.from_documents(
        documents=documents,
        embedding=embeddings,
        cosmos_client=cosmos_client,
        database_name=database_name,
        container_name=container_name,
        vector_embedding_policy=vector_embedding_policy,
        indexing_policy=indexing_policy,
        cosmos_container_properties=cosmos_container_properties,
        cosmos_database_properties={},
        vector_search_fields=vector_search_fields
    )
    return vectorstore


def encode_chunk(
    documents,
    embeddings,
    database_name,
    container_name,
    chunk_size=1000,
    chunk_overlap=200,
):
    print("="*20 + "ENCODING CHUNK" + "="*20)
    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    texts = text_splitter.split_documents(documents)
    cleaned_texts = replace_t_with_space(texts)
    vectorstore = create_vectorstore(cleaned_texts, embeddings, database_name, container_name)
    #
    # vectorstore = FAISS.from_documents(cleaned_texts, embeddings)
    # vectorstore.save_local("embedding/chunks_vector_store")
    return vectorstore


def encode_doc_summary(chapters, embeddings, database_name, container_name):
    print("="*20 + "ENCODING DOCUMENT SUMMARY" + "="*20)
    chapter_summaries = []
    # Iterate over each chapter in the chapters list
    for chapter in chapters:
        summary = create_chapter_summary(chapter, llm)
        chapter_summaries.append(summary)
    #
    docs_json = [doc.model_dump() for doc in chapter_summaries]
    with open("processed_docs/hp/chapter_summary.json", "w", encoding="utf-8") as f:
        json.dump(docs_json, f, ensure_ascii=False, indent=4)
    #
    # vectorstore = FAISS.from_documents(chapter_summaries, embeddings)
    # vectorstore.save_local("embedding/chapter_summaries_vector_store")
    #
    vectorstore = create_vectorstore(chapter_summaries, embeddings, database_name, container_name)
    return vectorstore


def encode_quote(book_quotes_list, embeddings, database_name, container_name):
    print("="*20 + "ENCODING DOCUMENT QUOTE" + "="*20)
    # vectorstore = FAISS.from_documents(book_quotes_list, embeddings)
    # vectorstore.save_local("embedding/book_quotes_vectorstore")
    #
    vectorstore = create_vectorstore(book_quotes_list, embeddings, database_name, container_name)
    return vectorstore


def encode_bm25(documents):
    print("="*20 + "ENCODING WORDS" + "="*20)
    # doc_content = [doc.page_content for doc in documents]
    # tokenized_docs = [doc.lower().split() for doc in doc_content]

    bm25_retriever = BM25Retriever.from_documents(
        documents, preprocess_func=tokenize, k=10
    )
    with open("embedding/bm25.pkl", "wb") as f:
        pickle.dump(bm25_retriever, f, protocol=pickle.HIGHEST_PROTOCOL)
    return


if __name__ == "__main__":
    doc_path = "docs/hp/Harry Potter - Book 1 - The Sorcerers Stone.pdf"
    process_quote = True
    process_chapter = True
    #
    document_cleaned, book_quotes_list, chapters = process_pdf_file(
        doc_path, process_quote, process_chapter
    )
    llm, embeddings = get_tools()
    chunk_vectorstore = encode_chunk(document_cleaned, embeddings, "rag-agent", "chunk-embedding")
    encode_bm25(document_cleaned)
    if process_chapter:
        chapter_vectorstore = encode_doc_summary(chapters, embeddings, "rag-agent", "chapter-embedding")
    if process_quote:
        quote_vectorstore = encode_quote(book_quotes_list, embeddings, "rag-agent", "quote-embedding")
