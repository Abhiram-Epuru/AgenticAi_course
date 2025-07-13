import os
import streamlit as st
from io import BytesIO
import PyPDF2
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores.faiss import FAISS
from langchain_core.runnables import RunnableMap, RunnableLambda, RunnablePassthrough
from langchain_core.prompts import PromptTemplate

load_dotenv()

st.set_page_config(page_title="Legal Document Review",layout="centered")
st.title("Legal Document Review (with LCEL and FIASS Vector Store)")

st.sidebar.header('API COnfiguration')
api_key = st.sidebar.text_input("Google Generative AI API Key", type="password")
if not api_key:
    st.sidebar.warning("Please enter your Google Generative AI API Key to proceed.")
os.environ["GOOGLE_GENERATIVE_AI_API_KEY"] = api_key

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "docs" not in st.session_state:
    st.session_state.docs = None

uploaded_file = st.file_uploader("Upload a PDF document", type="pdf")
if uploaded_file :
    try:
        reader = PyPDF2.PdfReader(BytesIO(uploaded_file.read()))
        raw_text = " ".join([page.extract_text()or "" for page in reader.pages])
        st.text_area("Extracted Text Preview", raw_text[:1000], height=200)

        splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        chunks= splitter.create_documents([raw_text])
        st.session_state.docs = chunks

        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001",google_api_key=api_key)
        vectorstore = FAISS.from_documents(chunks, embeddings)
        st.session_state.vectorstore = vectorstore
        st.success("Document processed and vector store created successfully!")
    except Exception as e:
        st.error(f"Error processing the document: {e}")

if st.session_state.vectorstore:
    st.subheader("Ask a Question about the Document")
    question = st.text_input("Ask a legal question from document:")
    
    if st.button('ASK'):
        retreiver=st.session_state.vectorstore.as_retriever(search_kwargs={"k": 4})
        prompt=PromptTemplate.from_template(
             """You are a legal assistant. Use the context to answer the user's question clearly.          
                Context:
                {context}
                Question: {question}
                Answer:""" 
        )
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash",google_api_key=api_key, temperature=0.2)
        rag_chain=(
            {"context": retreiver, "question": RunnablePassthrough()}
            |prompt
            |llm
        )
        with st.spinner("Generating answer..."):
            result=rag_chain.invoke(question)
        st.markdown("### Answer:")
        st.write(result.content)
    if st.button('Generate Summary'):
        retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 4})
        docs = retriever.invoke("summarize this document")
        context = "\n\n".join([doc.page_content for doc in docs])

        sprompt = PromptTemplate.from_template(
            """Summarize the key points, obligations, and time durations of this legal document:
            {context}
            Brief Summary:"""
        )
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash",google_api_key=api_key, temperature=0.2)

        summary_chain = (
        RunnableMap({"context": lambda _: context})
        | sprompt
        | llm
        )

        with st.spinner("Generating summary..."):
            summary_result = summary_chain.invoke({})
        st.markdown("### Summary:")
        st.write(summary_result.content)
