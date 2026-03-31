import os
import requests
from bs4 import BeautifulSoup
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI  # 只用於生成
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEmbeddings  # 新增：本地嵌入

# API key (僅用於生成；嵌入現在本地)
API_KEY = "yoyr-api-key"
GENERATION_MODEL = "meta-llama/llama-3.1-8b-instruct"
BASE_URL = "https://openrouter.ai/api/v1"

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def load_documents(source, source_type="text"):
    """
    Load documents from a file or web URL.
    """
    if source_type == "text":
        with open(source, 'r', encoding='utf-8') as f:
            text = f.read().strip()
        docs = [Document(page_content=text, metadata={"source": source})]
    elif source_type == "web":
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"}
        response = requests.get(source, headers=headers)
        if response.status_code != 200:
            raise ValueError(f"Failed to fetch web page: {response.status_code}")
        
        soup = BeautifulSoup(response.text, 'html.parser')
        main_content = soup.find('div', class_='guide-content') or soup.find('article') or soup.find('body')
        text = main_content.get_text(separator='\n', strip=True) if main_content else ""
        
        if not text.strip():
            raise ValueError("Extracted web content is empty. The page may be dynamic or protected.")
        
        docs = [Document(page_content=text, metadata={"source": source})]
    else:
        raise ValueError("Unsupported source_type. Use 'text' or 'web'.")
    
    if not docs or not docs[0].page_content.strip():
        raise ValueError("Loaded documents are empty or contain no text content.")
    return docs

def split_documents(documents, chunk_size=1000, chunk_overlap=200):
    """
    Split documents into smaller chunks for embedding.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
    )
    splits = splitter.split_documents(documents)
    if not splits:
        raise ValueError("Document splitting resulted in no chunks. Check document content.")
    return splits

def create_vector_store(documents):
    """
    Create a vector store from documents using local embeddings.
    """
    splits = split_documents(documents)
    splits = [s for s in splits if len(s.page_content.strip()) > 50]  # 過濾微小塊
    if not splits:
        raise ValueError("After filtering, no valid chunks remain. Check document content.")
    
    # 使用本地嵌入創建 FAISS
    vector_store = FAISS.from_documents(splits, embeddings)
    
    # 除錯：檢查嵌入維度
    sample_embedding = embeddings.embed_query("test")
    print(f"Embedding dimension: {len(sample_embedding)}")
    
    return vector_store

def rag_query(query, vector_store, k=4, temperature=0.7, filter=None, language="Spanish"):
    """
    Perform a RAG query.
    """
    search_kwargs = {"k": k}
    if filter:
        search_kwargs["filter"] = filter  # 例如，{"scenario": "restaurant"}
    retriever = vector_store.as_retriever(search_kwargs=search_kwargs)
    
    # 生成模型（使用 OpenRouter；若失敗，可切換本地）
    llm = ChatOpenAI(
        model=GENERATION_MODEL,
        base_url=BASE_URL,
        api_key=API_KEY,
        temperature=temperature,
    )
    
    prompt_template = """
    You are a helpful {language} language tutor in a conversation practice session.
    Use the following context to guide your response:
    {context}
    
    User Query: {question}
    
    Respond in the target language ({language}) first, then provide an English translation in parentheses or in a separate `translation` field.
    Keep it conversational: correct errors politely, suggest 1-2 vocabulary/grammar tips (put tips in English).
    If context is irrelevant, provide a general helpful answer.
    """

    # Inject the target language into the prompt template so the model knows which language to use
    prompt_text = prompt_template.format(language=language)
    prompt = ChatPromptTemplate.from_template(prompt_text)
    
    def format_docs(docs):
        formatted = "\n\n".join(doc.page_content for doc in docs)
        if len(formatted) > 4000:
            formatted = formatted[:4000] + "... (truncated)"
        return formatted
    
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    try:
        # The chain expects a question input; provide the raw query string.
        return chain.invoke(query)
    except Exception as e:
        print(f"Generation error (check API key/credits): {str(e)}")
        # 備用：簡單回應（無 LLM）
        return f"Sorry, unable to generate response due to API issue. Context summary: {format_docs(retriever.invoke(query))[:500]}..."

def load_multiple_documents(sources):
    all_docs = []
    for src, src_type, metadata in sources:
        docs = load_documents(src, src_type)
        for doc in docs:
            doc.metadata.update(metadata)
        all_docs.extend(docs)
    return all_docs

class RAGSystem:
    def __init__(self, sources=None):
        if sources is None:
            sources = [
                ("language_data.txt", "text", {"topic": "general"}),
                ("https://www.spanishdict.com/guide/spanish-present-tense-forms", "web", {"topic": "grammar", "subtopic": "present_tense"}),
                ("https://www.spanishdict.com/examples/restaurant", "web", {"scenario": "restaurant", "topic": "conversation"}),
                # 可以新增更多來源，例如：
                # ("https://www.spanishdict.com/guide/spanish-past-tense", "web", {"topic": "grammar", "subtopic": "past_tense"}),
            ]  # 預設來源清單（從 __main__ 範例衍生）
        docs = load_multiple_documents(sources)
        self.vector_store = create_vector_store(docs)
        self.vector_store.save_local("faiss_index")
    
    def query(self, query, k=4, temperature=0.7, filter=None):
        # Default language to English/Spanish context unless caller supplies via filter or args.
        # If caller provided a filter with 'language', extract it; otherwise default to Spanish.
        lang = None
        if isinstance(filter, dict) and 'language' in filter:
            lang = filter.get('language')
        if lang is None:
            lang = 'Spanish'
        return rag_query(query, self.vector_store, k, temperature, filter, language=lang)
        
# Example usage
if __name__ == "__main__":
    local_file = "language_data.txt"
    url = "https://www.spanishdict.com/guide/spanish-present-tense-forms"
    
    if not os.path.exists(local_file) or os.path.getsize(local_file) == 0:
        print(f"The local file {local_file} does not exist or is empty; downloading from the web...")
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"}
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            main_content = soup.find('body') or soup
            clean_text = main_content.get_text(separator='\n', strip=True)
            with open(local_file, "w", encoding="utf-8") as f:
                f.write(clean_text)
            print(f"The data has been stored to {local_file}")
        else:
            raise ValueError(f"Download failed: {response.status_code}")
    
    sources = [
    ("language_data.txt", "text", {"topic": "general"}),
    ("https://www.spanishdict.com/guide/spanish-present-tense-forms", "web", {"topic": "grammar", "subtopic": "present_tense"}),
    ("https://www.spanishdict.com/examples/restaurant", "web", {"scenario": "restaurant", "topic": "conversation"}),
    ]

    docs = load_documents(local_file, source_type="text")
    vector_store = create_vector_store(docs)
    
    vector_store.save_local("faiss_index")
    
    query = "Explain the present tense in Spanish."
    response = rag_query(query, vector_store)
    print("RAG Response:")
    print(response)
