from typing import List
import streamlit as st
import tempfile
import os
import torch
import chromadb
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from langchain_chroma import Chroma
from langchain_huggingface.llms import HuggingFacePipeline
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from transformers import BitsAndBytesConfig
from utils.logging_utils import logger
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
#from dotenv import load_dotenv
#load_dotenv() # Load variables from .env

# Session state initialization
if 'rag_chain' not in st.session_state:
    st.session_state.rag_chain = None
if 'models_loaded' not in st.session_state:
    st.session_state.models_loaded = False
if 'embeddings' not in st.session_state:
    st.session_state.embeddings = None
if 'llm' not in st.session_state:
    st.session_state.llm = None

# Functions
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="bkai-foundation-models/vietnamese-bi-encoder")

@st.cache_resource  
def load_llm():
    MODEL_NAME = "lmsys/vicuna-7b-v1.5"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,  # Hoặc load_in_8bit=True
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4"  # nf4 là lựa chọn tốt cho mô hình lớn
    )

    # Load model với quantization
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto"
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        pad_token_id=tokenizer.eos_token_id,
        device_map="auto"
    )

    # model = AutoModelForCausalLM.from_pretrained(
    #     MODEL_NAME,
    #     torch_dtype=torch.float16,  # hoặc torch.float16 nếu muốn tiết kiệm memory
    #     low_cpu_mem_usage=True,
    #     device_map="cpu"  # Explicitly set to CPU
    # )
    
    # tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # # Create pipeline with CPU device
    # model_pipeline = pipeline(
    #     "text-generation",
    #     model=model,
    #     tokenizer=tokenizer,
    #     max_new_tokens=512,
    #     pad_token_id=tokenizer.eos_token_id,
    #     device="cpu"  # Explicitly set device to CPU
    # )
    
    return HuggingFacePipeline(pipeline=model_pipeline)

def build_custom_rag_prompt():
    # Define a custom prompt for the RAG system
    
    template = """
    <Context>
    {context}
    </Context>

    Dựa vào thông tin trong phần Context ở trên, hãy trả lời câu hỏi sau một cách chính xác và đầy đủ.
    Nếu không thể tìm thấy thông tin để trả lời trong Context, hãy nói rõ rằng bạn không có đủ thông tin để trả lời.
    Trả lời bằng tiếng Việt và đảm bảo các thông tin được trích dẫn từ nguồn đã cung cấp.

    Question: {question}
    
    Answer:

    Trích dẫn:
    """
    
    prompt = ChatPromptTemplate.from_template(template)
    return prompt

def build_rag_prompt_v1():
    # Define a custom prompt for the RAG system
    
    template = """
    Bạn là một trợ lý cho các nhiệm vụ trả lời câu hỏi. Hãy sử dụng các phần ngữ cảnh được truy xuất sau đây để trả lời câu hỏi. Nếu bạn không biết câu trả lời, chỉ cần nói rằng bạn không biết. Sử dụng tối đa ba câu và giữ cho câu trả lời ngắn gọn.
    Question: {question} 
    Context: {context} 
    Answer:
    """
    
    prompt = ChatPromptTemplate.from_template(template)
    return prompt
def get_chroma_client(allow_reset=False):
    """Get a Chroma client for vector database operations."""
    # Use PersistentClient for persistent storage
    return chromadb.PersistentClient(settings=chromadb.Settings(allow_reset=allow_reset))

def process_pdf(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, prefix = uploaded_file.name, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name
    
    loader = PyPDFLoader(tmp_file_path)
    documents = loader.load()
    
    semantic_splitter = SemanticChunker(
        embeddings=st.session_state.embeddings,
        buffer_size=1,
        breakpoint_threshold_type="percentile", 
        breakpoint_threshold_amount=95,
        min_chunk_size=500,
        add_start_index=True
    )
    
    docs = semantic_splitter.split_documents(documents)
    chroma_client = get_chroma_client(allow_reset=True)
    chroma_client.reset() # empties and completely resets the database. This is destructive and not reversible.
    
    vector_db = Chroma.from_documents(
        documents=docs, 
        embedding=st.session_state.embeddings,
        client=chroma_client)
    retriever = vector_db.as_retriever()
    
    #prompt = build_custom_rag_prompt()
    prompt = hub.pull("rlm/rag-prompt")
    
    def format_docs(docs):
        logger.info(f"**Debug: Retrieved {len(docs)} chunks:**")
        for i, doc in enumerate(docs):
            # Extract metadata if available
            # Assuming each doc has metadata with 'page' and 'source'
            page_num = doc.metadata.get('page') + 1 if 'page' in doc.metadata else -1
            source = doc.metadata.get('source', 'document')
            file_name = os.path.basename(source) if isinstance(source, str) else 'unknown'

            logger.info(f"""
            ([reference-{i+1}] Page {page_num} - Source: {file_name})
            {doc.page_content}""")
        result = "\n\n".join(doc.page_content for doc in docs)
        #logger.info(result)
        return result
    
    def format_docs_with_citation(docs: List[Document])-> str:
        logger.info(f"**Debug: Retrieved {len(docs)} chunks:**")
        formatted_docs = []
        for i, doc in enumerate(docs):
            # Extract metadata if available
            # Assuming each doc has metadata with 'page' and 'source'
            page_num = doc.metadata.get('page')
            source = doc.metadata.get('source', 'document')
            formatted_docs.append(f"""
Trích dẫn: {source} (Trang {page_num})
Nội dung: {doc.page_content}""")
            
        result = "\n\n".join(formatted_docs)
        return result
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt 
        | st.session_state.llm
        | StrOutputParser()
    )
    
    os.unlink(tmp_file_path)
    return rag_chain, len(docs)

# UI
def main():
    st.set_page_config(page_title="PDF RAG Assistant", layout="wide")
    st.title("PDF RAG Assistant")

    st.markdown("""
    **Ứng dụng AI giúp bạn hỏi đáp trực tiếp với nội dung tài liệu PDF bằng tiếng Việt**

    **Cách sử dụng đơn giản:**
    1. **Upload PDF** → Chọn file PDF từ máy tính và nhấn "Xử lý PDF"  
    2. **Đặt câu hỏi** → Nhập câu hỏi về nội dung tài liệu và nhận câu trả lời ngay lập tức

    ---
    """)

    # Load models
    if not st.session_state.models_loaded:
        st.info("Đang tải models...")
        st.session_state.embeddings = load_embeddings()
        st.session_state.llm = load_llm()
        st.session_state.models_loaded = True
        st.success("Models đã sẵn sàng!")
        st.rerun()

    # Upload PDF
    uploaded_file = st.file_uploader("Upload file PDF", type="pdf")
    if uploaded_file and st.button("Xử lý PDF"):
        with st.spinner("Đang xử lý..."):
            st.session_state.rag_chain, num_chunks = process_pdf(uploaded_file)
            st.success(f"Hoàn thành! {num_chunks} chunks")

    # Q&A
    if st.session_state.rag_chain:
        question = st.text_input("Đặt câu hỏi:")
        if question:
            with st.spinner("Đang trả lời..."):
                output = st.session_state.rag_chain.invoke(question)
                answer = output.split('Answer:')[1].strip() if 'Answer:' in output else output.strip()
                st.write("**Trả lời:**")
                st.write(answer)

if __name__ == "__main__":
    main()
