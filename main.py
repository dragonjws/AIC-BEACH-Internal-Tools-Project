import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
import time
from typing import List, Dict, Any
import chromadb
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
import numpy as np
import boto3
import json
load_dotenv()

def load_and_chunk_documents(documents):
    policy_documents = documents
    #load documents into policy_documents

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 200,
        chunk_overlap=50, #give context
        length_function = len,
        separators = ["\\n", "\n", " ", ""], 
    )

    all_chunks = []

    for doc in policy_documents:
        
        #change "content" to whatever label we give it
        chunks = text_splitter.split_text(doc["content"]) 
        for i, chunk in enumerate(chunks):
            all_chunks.append({
                #append dictionary values as appropriate
                "id": f"{doc["title"]}_{i}", #use title_index here to create unique chunks
                "title": doc["title"],
                "content": chunk,
                #"category": "some category",
                "source_doc": doc["source_doc"],
            }) 
    return all_chunks

def setup_vector_database(chunks: List[Dict], num_new_files):
    client = chromadb.PersistentClient(path = "./my_local_db")

    #st_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    collection = client.get_or_create_collection(
            name = "test_docs",
            metadata = {"hnsw:space": "cosine"} #calculate vector simularities using cosine metric
        )
    if len(chunks) ==0:
        print("No chunks passed in. DB initialized")
        return collection

    print(f"adding {num_new_files} new chunks...")
    ids = [chunk["id"] for chunk in chunks]
    documents = [chunk["content"] for chunk in chunks]
    metadatas = [{
        "title": chunk["title"], 
        #"category":chunk["category"],
        "source_doc": chunk["source_doc"]} for chunk in chunks]
        
    #skips items with the same ids, only adds new ones. safety net because the new docs should already be handled before chunking
    collection.upsert(
            ids = ids,
            documents = documents,
            metadatas = metadatas
        )
    return collection

def process_user_query(query:str):
    model = SentenceTransformer('all-MiniLM-L6-v2')

    cleaned_query = query.lower().strip()
    query_embedding = model.encode([cleaned_query])

    return model, query_embedding[0]

def search_vector_database(collection, query_embedding, top_k: int = 3):
    
    #take your collectino of vectorized texts, and compare it to your query. Add the top 3 closest matches
    results = collection.query(
        query_embeddings = [query_embedding.tolist()],
        n_results= top_k #how many results we want to return
    )
    search_results = []

    #looop through each of the 3 top similarity docs then add all of the appropriate data to each element in the search_results list.

    #chromadb gives a dictionary of separate lists: one with all the ids, another with all the distances, etc. Zip takes all the elements in each list with the same index.
    for i, (doc_id, distance, content, metadata) in enumerate(zip(
        results["ids"][0],
        results['distances'][0], 
        results['documents'][0], 
        results['metadatas'][0]
    )):
        similarity = 1- distance #convert distance to similarity

        search_results.append({
            'id': doc_id,
            'content': content,
            'metadata': metadata,
            'similarity': similarity
        })

    return search_results

#creates gift wrapped prompt to feed LLM: giving it your question, the appropriate context (top 3 sources), and prompt as to what to do
def augment_prompt_with_context(query: str, search_results: List[Dict]) -> str:
    context_parts = []
    for i, result in enumerate(search_results, 1):


        context_parts.append(f'Source {i}: {result["metadata"]["title"]}\n{result["content"]}')

        context = "\n\n".join(context_parts)

        augmented_prompt = f"""
        Based on the following company policies, answer the user's questions.

        POLICES:
        {context}

        QUESTION: {query}

        Please provide a clear, accurate answer based on the policies above.
        If the information is not available in the policies, say so.
        Include relevant policy details and any limitations or requirements.
        """ 

        print(f'context sources: {[result["metadata"]["title"] for result in search_results]}')

    return augmented_prompt 




def generate_response(docs, query) -> str:
    class ModelResponse(BaseModel):
        direct_answer: str = Field(description = "A clear synthesis of the answer.")
        source: str = Field(description = "The content from the chunk of the best source along with its source_id") #consider making this a direct variable plug rather than an LLM thing
        source_analysis: str = Field(description = "Detailed reasoning for the why this is the best source that answers the query.")
        citation: str = Field(description="Exact citations using [Title | Path: source_path | id: id].")
    parser = PydanticOutputParser(pydantic_object = ModelResponse)

    prompt_template= """
    ### ROLE
    You are a professional AI Librarian. Your expertise is in analyzing research documents and identifying the most relevant information for a user's specific query. 

    ### TASK
    I will provide you with a USER QUESTION {query} and three context chunks {docs} retrieved from our internal database. Your goal is to:
    1. ANSWER the question clearly and concisely.
    2. RANK and IDENTIFY which of the three sources provided the most relevant information.
    3. EXPLAIN WHY those specific chunks were chosen as the "best fit" for the research topic.
    4. CITE the source for every claim made using the format: [Title | Path: source_path].

    ### CONSTRAINTS
    - Use ONLY the provided context to answer. If the answer isn't there, say: "I am sorry, but the current library sources do not contain enough information to answer this."
    - Do not use outside knowledge or "hallucinate" details.
    - If sources contradict each other, highlight the discrepancy to the researcher.

    {format_instructions}
    """
    llm = ChatOpenAI(model = "gpt-4o-mini", temperature = 0.0)

    prompt = PromptTemplate(
        input_variables = ["docs",
                           "query",
                           "format_instructions"],
        template = prompt_template
    )
    chain = prompt | llm | parser
    
    sources = "\n\n".join([
    f"SOURCE {i}:\nTitle: {res['metadata']['title']}\nPath: {res['metadata']['source_doc']}\nContent: {res['content']}"
    for i, res in enumerate(docs, 1)
    ])

    response = chain.invoke({
        "docs": sources,
        "query": query,
        "format_instructions": parser.get_format_instructions(),
    })
    return response


def run_complete_rag_pipeline(documents):
    """
    Pipeline:
    1) Load doc and chunk
    2) Setup Vector database 
    3) Process user input
    4) Search for top 3 docs relating to query
    5) Return LLM with a prompt with appropriate context
    6) Generate a response
    """
    query = input("What do you want to chat about: ")

    collection = setup_vector_database([], 0) #load nothing in. just set it up if you haven't already
    print("initialized database successfully")
    print(f"Loaded {len(documents)} document(s) into the RAG pipeline.")
    if not documents:
        print("No documents loaded from S3. Please verify bucket contents and AWS credentials.")
        return
    
    #now we check which files have already been added as a chunk in the vector db
    existing_metadatas = collection.get(include = ["metadatas"])["metadatas"]
    already_processed_files = {m["title"] for m in existing_metadatas}
    files_to_process = [d for d in documents if d['title'] not in already_processed_files]
    num_new_files = len(files_to_process)
    if files_to_process:
        print("new files detected")
        chunks = load_and_chunk_documents(files_to_process)
        collection = setup_vector_database(chunks, num_new_files)
        print("New vectors added successfully")

    else:
        print("no new files")

    #now we account for what's already stored in our database

    model, query_embedding = process_user_query(query)
    print("Processed query successfully")

    search_results = search_vector_database(collection, query_embedding)
    print("Searched vector database successfully")


    #augmented_prompt = augment_prompt_with_context(query, search_results)

    response = generate_response(search_results, query)
    print("Response generated successfully")


    print("-------------Direct Answer--------------")
    print(response.direct_answer)
    print("--------------Best Source----------")
    print(response.source)
    print("------------Source Analysis------------")
    print(response.source_analysis)
    print("-------------Citation----------")
    print(response.citation)





def load_documents(folder_name):
    documents = []
    i = 0
    for e in os.scandir(path = folder_name):
        if e.is_file():
            with open(e.path, "r") as f:
                documents.append({
                    "id": f"{i}",
                    "title": e.name,
                    "content": f.read(),
                    "source_doc": e.path,
                    })
                i += 1
    return documents

def load_documents_from_s3(bucket_name):
    """
    Load text files from an S3 bucket and return them as documents.
    Saves the documents to a JSON file for future use.
    """
    json_filename = f"{bucket_name}_documents.json"
    
    # Check if JSON file already exists
    if os.path.exists(json_filename):
        print(f"Loading documents from existing JSON file: {json_filename}")
        with open(json_filename, 'r', encoding='utf-8') as f:
            documents = json.load(f)
        return documents
    
    s3_client = boto3.client('s3')
    documents = []
    i = 0
    
    try:
        # List all objects in the bucket
        response = s3_client.list_objects_v2(Bucket=bucket_name)
        
        if 'Contents' in response:
            for obj in response['Contents']:
                key = obj['Key']
                # Only process text files (you can modify this filter as needed)
                if key.endswith(('.txt', '.md', '.json')):
                    try:
                        # Get the object from S3
                        s3_response = s3_client.get_object(Bucket=bucket_name, Key=key)
                        content = s3_response['Body'].read().decode('utf-8')
                        
                        documents.append({
                            "id": f"{i}",
                            "title": key.split('/')[-1],  # Use filename as title
                            "content": content,
                            "source_doc": f"s3://{bucket_name}/{key}",
                        })
                        i += 1
                        print(f"Loaded document: {key}")
                    except Exception as e:
                        print(f"Error loading {key}: {e}")
        
        # Save documents to JSON file
        with open(json_filename, 'w', encoding='utf-8') as f:
            json.dump(documents, f, indent=2, ensure_ascii=False)
        print(f"Saved {len(documents)} documents to {json_filename}")
        
    except Exception as e:
        print(f"Error accessing S3 bucket {bucket_name}: {e}")
    
    return documents


if __name__ == "__main__":
    bucket_name = "updatedbucketssss"
    documents = load_documents_from_s3(bucket_name)
    #try:
    run_complete_rag_pipeline(documents)
    #except Exception as e: 
    #    print("error in demo: {e}")
    
