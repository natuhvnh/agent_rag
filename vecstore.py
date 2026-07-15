import os
from azure.cosmos import CosmosClient, PartitionKey
from langchain_azure_cosmosdb import AzureCosmosDBNoSqlVectorSearch
from dotenv import load_dotenv

load_dotenv()

# Define policies ONCE
VECTOR_EMBEDDING_POLICY = {
    "vectorEmbeddings": [
        {
            "path": "/vector_embedding",
            "dataType": "float32",
            "distanceFunction": "cosine",
            "dimensions": 1536,
        }
    ]
}

INDEXING_POLICY = {
    "indexingMode": "consistent",
    "automatic": True,
    "includedPaths": [{"path": "/*"}],
    "excludedPaths": [{"path": '/"_etag"/?'}],
    "vectorIndexes": [{"path": "/vector_embedding", "type": "diskANN"}],
}


def get_vector_store(embeddings, db_name, container_name):
    """Returns a configured Cosmos DB Vector Store instance."""
    endpoint = os.environ.get("cosmos_url")
    key = os.environ.get("cosmos_key")
    client = CosmosClient(url=endpoint, credential=key)
    vector_search_fields = {"text_field": "text", "embedding_field": "vector_embedding"}
    cosmos_container_properties = {"partition_key": PartitionKey(path="/id")}

    return AzureCosmosDBNoSqlVectorSearch(
        cosmos_client=client,
        embedding=embeddings,
        database_name=db_name,
        container_name=container_name,
        vector_embedding_policy=VECTOR_EMBEDDING_POLICY,
        indexing_policy=INDEXING_POLICY,
        cosmos_container_properties=cosmos_container_properties,
        cosmos_database_properties={},
        vector_search_fields=vector_search_fields,
    )
