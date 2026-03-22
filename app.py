import streamlit as st
from jinja2 import Template
from dotenv import load_dotenv

# Embeddings - move to community
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
# Google model - use genai instead of deprecated palm
from langchain_google_genai import ChatGoogleGenerativeAI
# Chains - still in main langchain
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_groq import ChatGroq
# Vector store - move to community
# from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma
# Document loaders - move to community
from langchain_community.document_loaders import PyPDFLoader

from PyPDF2 import PdfReader, PdfWriter
from tempfile import NamedTemporaryFile
import base64
from htmlTemplates import expander_css, css, bot_template, user_template
import numpy as np
import time

@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': False}
    )

def process_file(pdf_file):
    model_name = "sentence-transformers/all-mpnet-base-v2"
    model_kwargs = {'device': 'cpu'} # Use 'cuda' if you have a compatible GPU
    encode_kwargs = {'normalize_embeddings': False}
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0 )
    # Initialize the HuggingFaceEmbeddings class
    llm=ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct",api_key=st.secrets["GROQ_API_KEY"])

    hf = load_embeddings()
    loader = PyPDFLoader(pdf_file)
    pages = loader.load_and_split()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.split_documents(pages)
    vector_store = Chroma(
            collection_name="foo",
            embedding_function=hf)
    vector_store.add_documents(documents=docs)
    # vector_store=Chroma.from_documents(documents=docs,embedding_function=hf)
    retriever = vector_store.as_retriever(
    search_type="similarity", # or "mmr" for diversity
    search_kwargs={"k": 3}   # number of documents to retrieve
)
    return ConversationalRetrievalChain.from_llm(llm, retriever,return_source_documents=True)
# Task 6: Method for Handling User Input
def get_answer(ques):
    answer=ques
    if "qa" in st.session_state and "chat_history" in st.session_state:
        answer=st.session_state.qa.invoke({"question":ques,"chat_history": st.session_state.chat_history})
        st.session_state.pdf.write(dict(answer["source_documents"][0])["metadata"]["page"])
        st.session_state.pgn=dict(answer["source_documents"][0])["metadata"]["page"]
        print(st.session_state.chat_history)
        return answer["answer"]
    return answer

def main():
    
    # Task 3: Create Web-page Layout
    
    load_dotenv()
    st.set_page_config(layout="wide",page_icon=":books:",page_title="Interactive Reader")
    chat,pdf=st.columns(2)
    # st.html(body=css)
    st.session_state.pdf_doc=None
    st.session_state.chat=chat
    st.session_state.pdf=pdf
    st.session_state.chat.title("Intreactive Reader :books:")
    st.session_state.chat.write("Ask question from the pdf.Ask question from the pdf.Ask question from the pdf.Ask question from the pdf.Ask question from the pdf.")
    if "chat_history" not in st.session_state:
        st.session_state.chat_history=[]
    curr=st.session_state.chat.text_input("Ask something ....","This is placeholder",key="placeholder")
    if "pgn" not in st.session_state:
        st.session_state.pgn=0
    with st.session_state.chat.expander("chat",expanded=True):
        # print(history)
        answer=get_answer(curr)
        if st.session_state.pdf_doc is not None:
            with NamedTemporaryFile(suffix="pdf") as temp2:
                    temp2.write(st.session_state.pdf_doc.getvalue())
                    with open(temp2.name, "rb") as f:
                        base64_pdf = base64.b64encode(f.read()).decode('utf-8')

                        pdf_display = f'''<iframe src="data:application/pdf;base64,{base64_pdf}#page={st.session_state.pgn+1}"
                            width="100%" height="900" type="application/pdf" frameborder="0"></iframe>'''
                    
                        st.session_state.pdf.markdown(pdf_display, unsafe_allow_html=True)
        st.session_state.chat_history.append(((curr,answer)))
        st.markdown(body=expander_css,unsafe_allow_html=True)
        st.markdown(body=css,unsafe_allow_html=True)
        for item in st.session_state.chat_history[::-1][:1]:
            user_t=Template(user_template)
            user_msg=user_t.render({"MSG":item[0]})
            st.markdown(body=user_msg,unsafe_allow_html=True)
            bot_t=Template(bot_template)
            bot_t=bot_t.render({"MSG":item[1]})
            st.markdown(body=bot_t,unsafe_allow_html=True)

    # pdf.bar_chart(np.random.randn(50,3))
    st.session_state.chat.subheader("Your Documents.")
    st.session_state.pdf_doc=st.session_state.chat.file_uploader("Upload a PDF and click 'Process'")
    if st.session_state.chat.button(label="Process"):
        with st.session_state.chat.spinner("Processing"):
            with NamedTemporaryFile(suffix="pdf") as temp:
                temp.write(st.session_state.pdf_doc.getvalue())
                temp.seek(0)
                # pdf_f=PdfReader(temp)
                st.session_state.pdfname=temp.name
                print(temp.name)
                st.write(temp.name)
                st.session_state.qa=process_file(temp.name)
                # st.session_state.qa=qa
                st.markdown("processing done! ")

    # Task 5: Load and Process the PDF 
    

    
    # Task 7: Handle Query and Display Pages
    
       



if __name__ == '__main__':
    main()

