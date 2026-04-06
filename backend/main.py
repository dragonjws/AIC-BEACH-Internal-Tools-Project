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
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import uvicorn
from typing import Optional
from fastapi import UploadFile, File
import shutil


upload_directory = "/Users/khang/Desktop/beach_test_files"
client = None
collection = None
model = SentenceTransformer('all-MiniLM-L6-v2')

def initialize_db():
    global client, collection
    client = chromadb.PersistentClient(path="./my_local_db")
    collection = client.get_or_create_collection(
        name="test_docs",
        metadata={"hnsw:space": "cosine"}
    )

class ModelResponse(BaseModel):
        direct_answer: str = Field(description = "A clear synthesis of the answer.")
        source: List[str] = Field(description = "List of the content from the three chunks of the best sources along with their source_id") #consider making this a direct variable plug rather than an LLM thing
        source_analysis: str = Field(description = "Detailed reasoning for the why this is the best source that answers the query.")
        citation: str = Field(description="Exact citations using [Title | Path: source_path | id: id].")

class Document(BaseModel):
    id: Optional[str] = None
    title: Optional[str] = None
    content: Optional[str] = None
    source_doc: Optional[str] = None

class ChatRequest(BaseModel):
    query: str


app = FastAPI()

#put future deployment endpoint here
origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://0.0.0.0:5173",
    "http://localhost:8000"
]

#use cors to block unauthorized api requests
app.add_middleware(
    CORSMiddleware,
    allow_origins = origins,
    allow_credentials= True,
    allow_headers = ["*"],
    allow_methods = ["*"]
)

#take in document from user and add to database
@app.post("/documents")
async def add_document(file: UploadFile = File(...)):
    file_path = os.path.join(upload_directory, file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    documents = load_documents(upload_directory)
    upload_and_process_docs(documents)
    return {"filename": file.filename, "path": file_path}

@app.post("/chat")
async def chat_with_llm(request: ChatRequest):
    return(run_complete_rag_pipeline(request.query))


load_dotenv()

def load_and_chunk_documents(documents):
    policy_documents = documents
    #load documents into policy_documents

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap=200, #give context
        length_function = len,
        separators = ["\n\n", "\n", " ", ""], 
    )

    all_chunks = []

    for doc in policy_documents:
        
        #change "content" to whatever label we give it
        chunks = text_splitter.split_text(doc["content"]) 
        for i, chunk in enumerate(chunks):
            #place the title of each doc at the top in order to give context to the llm later on
            contextualized_content = f"Context: {doc["title"].replace(".txt","")} \nContent: {chunk}"
            all_chunks.append({
            

                #append dictionary values as appropriate
                "id": f"{doc['title']}_{i}", #use title_index here to create unique chunks
                "title": doc["title"],
                "content": contextualized_content,
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

def process_user_query(query:str, model):

    cleaned_query = query.lower().strip()
    query_embedding = model.encode([cleaned_query])

    return query_embedding[0]

def search_vector_database(collection, query_embedding, top_k: int = 10):
    
    #take your collectino of vectorized texts, and compare it to your query. Add the top 10 closest matches
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
    
    print(search_results[0]['content'])
    print(search_results[1]['content'])
    print(search_results[2]['content'])


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
    parser = PydanticOutputParser(pydantic_object = ModelResponse)

    prompt_template= """
    ### ROLE
    You are a professional AI Librarian.

    ### TASK
    I will provide you with a USER QUESTION {query} and context chunks {docs}.
    
    1. ANSWER the question clearly.
    2. You MUST provide exactly THREE (3) distinct sources in the 'source' list. 
    3. If there are multiple chunks provided in the context, pick the 3 most relevant ones and place them in the list as separate strings.
    4. Even if one source is very good, you MUST still provide two other supporting or contextual sources from the provided text to fill the 3 slots.

    ### CONSTRAINTS
    - Do not combine sources. 
    - Every entry in the 'source' list must be a raw string of text from the provided context.
    
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
    
    #explicitly give "real_filename"
    sources = "\n\n".join([
        f"SOURCE {i}:\n"
        f"REAL_FILENAME: {res['metadata']['title']}\n" # Give it the actual metadata title
        f"FILE_PATH: {res['metadata']['source_doc']}\n"
        f"TEXT_CONTENT: {res['content']}"
        for i, res in enumerate(docs, 1)
])

    response = chain.invoke({
        "docs": sources,
        "query": query,
        "format_instructions": parser.get_format_instructions(),
    })
    return response

def upload_and_process_docs(documents):
    collection = setup_vector_database([], 0) #load nothing in. just set it up if you haven't already
    print("initialized database successfully")
    
    #now we check which files have already been added as a chunk in the vector db 
    #rework this checking system
    
    data = collection.get(include = ["metadatas"])
    existing_metadatas = data.get("metadatas", [])

    already_processed_files = {m["title"] for m in existing_metadatas}
    files_to_process = [d for d in documents if d['title'] not in already_processed_files]
    num_new_files = len(files_to_process)
    if files_to_process:
        print("new files detected")
        chunks = load_and_chunk_documents(files_to_process)
        collection = setup_vector_database(chunks, len(chunks))
        print("New vectors added successfully")

    else:
        print("no new files")


def run_complete_rag_pipeline(query):
    """
    Pipeline:
    1) Load doc and chunk
    2) Setup Vector database 
    3) Process user input
    4) Search for top 3 docs relating to query
    5) Return LLM with a prompt with appropriate context
    6) Generate a response
    """

    query_embedding = process_user_query(query, model)
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
    
    return {
        "answer": response.direct_answer,
        "source_1": response.source[0],
        "source_2": response.source[1] if len(response.source) > 1 else "",
        "source_3": response.source[2] if len(response.source) > 2 else "",
        "analysis": response.source_analysis,
        "citation": response.citation
    }
    





def load_documents(folder_name):
    documents = []
    i = 0
    for e in os.scandir(path = folder_name):
        if e.is_file() and not e.name.startswith('.') and e.name.endswith('.txt'):
            with open(e.path, "r") as f:
                documents.append({
                    "id": f"{i}",
                    "title": e.name,
                    "content": f.read(),
                    "source_doc": e.path,
                    })
                i += 1
    return documents


if __name__ == "__main__": 
    db_path = "./my_local_db"
    db_exists = os.path.exists(db_path) and len(os.listdir(db_path)) > 0

    initialize_db()

    if not db_exists:
        print("No local database found. Initializing from folder...")
        initial_docs = load_documents(upload_directory)
        if initial_docs:
            upload_and_process_docs(initial_docs)
        else:
            print("Warning: No .txt files found.")
    else:
        print("Local database instance found. Skipping initial load.")

    uvicorn.run(app, host= "0.0.0.0", port=8000)
    #here would go any type of files/database

    #documents = load_documents(upload_directory)
    #run_complete_rag_pipeline(documents)

    
