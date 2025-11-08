import os
import fitz  # PyMuPDF
import streamlit as st
from dotenv import load_dotenv

# === 0. Corrigir conflito de bibliotecas OpenMP ===
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# === 1. Imports LangChain compatíveis com 0.1.0+ ===
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyMuPDFLoader

# === 2. Carregar variáveis de ambiente ===
load_dotenv()

st.set_page_config(page_title="ChatPDF com RAG", layout="wide")
st.title("📄 Chat de auxílio jurídico")

# === 3. Upload do PDF ===
pdf_file = st.file_uploader("Envie um PDF", type=["pdf"])

if pdf_file:
    os.makedirs("docs", exist_ok=True)
    file_path = f"docs/{pdf_file.name}"

    with open(file_path, "wb") as f:
        f.write(pdf_file.read())

    # === 4. Leitura e divisão dos textos ===
    loader = PyMuPDFLoader(file_path)
    docs = loader.load()

    splitter = CharacterTextSplitter(
        separator="\n", 
        chunk_size=1000, 
        chunk_overlap=100
    )
    texts = splitter.split_documents(docs)

    # === 5. Embeddings e vetorstore (FAISS) ===
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(texts, embeddings)

    # === 6. Criação do Retriever e Chain RAG ===
    retriever = vectorstore.as_retriever(
        search_type="similarity", 
        search_kwargs={"k": 4}
    )

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    rag_chain = RetrievalQA.from_chain_type(
        llm=llm, 
        retriever=retriever,
        return_source_documents=True
    )

    st.success("✅ PDF processado com sucesso! Agora você pode fazer perguntas.")

    # === 7. Interface de perguntas ===
    user_question = st.text_input("❓ Pergunte algo sobre o documento:")

    if user_question:
        with st.spinner("🔍 Consultando o conteúdo do documento..."):
            # Nova chamada com o método .invoke() (não .run)
            resposta = rag_chain.invoke({"query": user_question})

        st.markdown("### 🧠 Resposta:")
        st.write(resposta["result"])

        # Exibir fontes (opcional)
        with st.expander("📚 Fontes consultadas"):
            for i, doc in enumerate(resposta["source_documents"]):
                st.markdown(f"**Trecho {i+1}:**")
                st.write(doc.page_content)

