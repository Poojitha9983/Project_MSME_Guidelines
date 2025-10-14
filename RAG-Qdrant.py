import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_qdrant import QdrantVectorStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from qdrant_client import QdrantClient

#---setup openai api key for embeddings & llm---

load_dotenv()

os.environ["OPENAI_API_KEY"] = "YOUR OPENAI_API_KEY"

#---Qdrant cloud info---

qdrant_url ="YOUR QDRANT_CLOUD_URL"
qdrant_api_key ="YOUR QDRANT_API_KEY"
collection_name = "msme_guidelines_docs"

#---load pdf---
pdf_path = r"C:\Users\HP\Desktop\MSME_Documents\Guideline_Book.pdf"
loader = PyPDFLoader(pdf_path)
documents = loader.load()

#---split into chunks---
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(documents)
print(f"Total chunks created: {len(texts)}")

#---embeddings model---
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

#---upload to qdrant---
qdrant = QdrantVectorStore.from_documents(
    documents= texts,
    embedding=embeddings,
    url=qdrant_url,
    prefer_grpc=True,
    api_key=qdrant_api_key,
    collection_name=collection_name,
    batch_size=20,
    force_recreate=False,
)
print(f"Documents stored in Qdrant collection: {collection_name}")


#verify collection in qdrant
qdrant_client = QdrantClient(
    url=qdrant_url, 
   api_key=qdrant_api_key,
   timeout=120.0,
 )

print(qdrant_client.get_collections())

# Create retriever from the same vectorstore (MMR-based)
retriever = qdrant.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 6, "fetch_k": 20}
 )

query = input("\n Enter a semantic search query: ")
results = retriever.invoke(query)

print("\n Top MMR Semantic Search Results:\n")
for i, doc in enumerate(results, 1):
 print(f"Result {i}:\n{doc.page_content[:400]}...\n Source: {doc.metadata.get('source', 'N/A')}\n")

 # --- STEP 11: Optional LLM QA (RAG) ---
use_llm = input("\n Do you want GPT to answer your question? (y/n): ").lower()
if use_llm == "y":
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    qa_chain = RetrievalQA.from_chain_type(
       llm=llm,
       retriever=retriever,
        return_source_documents=True
     )

    question = input("\n Ask your question: ")
    result = qa_chain.invoke({"query": question})

    print("\n GPT Answer:\n", result["result"])
    print("\n Sources:")
    for src in result["source_documents"]:
       print("-", src.metadata.get("source", "N/A"))