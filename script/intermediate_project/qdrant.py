from qdrant_client import QdrantClient, models

def create_collection(vector_size, collection_name,  dataframe, ids):
    client = QdrantClient(host='localhost', port=6333)    
    # if client.collection_exists(collection_name):
    #     client.delete_collection(collection_name)
    if not client.collection_exists(collection_name):
        print("Creating collection...")
        client.create_collection(
            collection_name,
            vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE)
        )
        insert_embeddings(dataframe.to_numpy(), collection_name, ids)
    
def insert_embeddings(embeddings, collection_name, product_ids):
    client = QdrantClient(host='localhost', port=6333)
    ids = [i for i in range(len(embeddings))]
    print(f"Inserting {len(embeddings)} embeddings...")
    for i in range(0, len(embeddings), 200):
        chunk_embeddings = embeddings[i:i+200]
        chunk_parent_asin = product_ids[i:i+200]
        chunk_ids = ids[i:i+200]
        print(f"Inserting embeddings {i} to {i+len(chunk_embeddings)}...")
        operation_status = client.upsert(
            collection_name,
            wait=True,
            points=[
                models.PointStruct(
                    id = id,
                    vector = embedding,
                    payload={"product_id": str(product_id)}
                )
                for id, product_id, embedding in zip(chunk_ids, chunk_parent_asin, chunk_embeddings)
            ]
        )
    client.create_payload_index(
        collection_name=collection_name,
        field_name="product_id",
        field_schema="keyword")
    return operation_status.status == models.UpdateStatus.COMPLETED

def search_similar_products(query_embedding, collection_name, top_k=5, ids=[]):

    
    client = QdrantClient(host='localhost', port=6333)
    if ids != []:
        response = client.search(
            collection_name,
            query_vector=query_embedding,
            limit=top_k,
            query_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="product_id",
                        match=models.MatchAny(any=ids),
                    )
                ]
            )
        )
    else:
        response = client.search(
            collection_name,
            query_vector=query_embedding,
            limit=top_k
        )
    return response

