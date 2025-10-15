import os
import uuid
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_qdrant import QdrantVectorStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from qdrant_client import QdrantClient

# ---Setup OpenAI API Key ---
load_dotenv()
os.environ["OPENAI_API_KEY"] ="YOUR OPENAI_API_KEY"

# ---Qdrant Cloud Info ---
qdrant_url ="YOUR QDRANT_URL"
qdrant_api_key ="YOUR QDRANT_API_KEY"
collection_name = "msme_guidelines_docs"

# ---Load PDF (single file only) ---
pdf_path = r"C:\Users\HP\Desktop\MSME_Documents\Guideline_Book.pdf"
loader = PyPDFLoader(pdf_path)
documents = loader.load()

# ---Split into chunks ---
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(documents)
print(f" Total chunks created: {len(texts)}")

# ---Generate a unique PDF ID (same for all chunks) ---
pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
pdf_id = f"{pdf_name}-{uuid.uuid4()}"  # Unique per PDF upload
print(f" Unique PDF ID: {pdf_id}")

# Add metadata to all chunks
for i, doc in enumerate(texts):
    doc.metadata["pdf_id"] = pdf_id
    doc.metadata["pdf_name"] = pdf_name
    doc.metadata["source"] = pdf_path
    doc.metadata["chunk_index"] = i

# ---Initialize Embeddings ---
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

# ---Upload to Qdrant ---
qdrant = QdrantVectorStore.from_documents(
    documents=texts,
    embedding=embeddings,
    url= qdrant_url,
    api_key= qdrant_api_key,
    prefer_grpc= True,
    collection_name=collection_name,
    batch_size=20,
    force_recreate=False,  # Keep existing documents
)

print(f" Uploaded {len(texts)} chunks for PDF: {pdf_name}")
print(f" Stored under PDF ID: {pdf_id}")

# ---Verify Qdrant Collections ---
qdrant_client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key, timeout=120.0)
print("\n Collections in Qdrant:")
print(qdrant_client.get_collections())

# ---Create Retriever (MMR-based Semantic Search) ---
retriever = qdrant.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 6, "fetch_k": 20},
)

# ---Query Qdrant ---
query =input("\n Enter a semantic search query: ")
results = retriever.invoke(query)

print("\n Top MMR Semantic Search Results:\n")
for i, doc in enumerate(results, 1):
    print(f"Result {i}:\n{doc.page_content[:400]}...\nSource: {doc.metadata.get('source', 'N/A')}\n")

# ---Optional GPT QA (RAG) ---
use_llm = input("\n Do you want GPT to answer your question? (y/n): ").lower()
if use_llm == "y":
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
    )

    question = input("\n Ask your question: ")
    result = qa_chain.invoke({"query": question})

    print("\n GPT Answer:\n", result["result"])
    print("\n Sources:")
    for src in result["source_documents"]:
        print("-", src.metadata.get("source", "N/A"))
