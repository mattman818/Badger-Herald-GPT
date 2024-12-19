#Badger Herald GPT

import streamlit as st
import langchain_community
import langchain_openai
from langchain_openai import ChatOpenAI, OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import OpenAIEmbeddings
import time
from datetime import datetime

st.title("The Badger Herald GPT")

# Define your API key
openai_api_key = st.secrets["OPENAI_API_KEY"]

# Remove the file uploader and directly process your text file
DEFAULT_TEXT_PATH = "default.txt"

# Initialize session state
if "conversation_chain" not in st.session_state:
    st.session_state.conversation_chain = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "last_activity" not in st.session_state:
    st.session_state.last_activity = None

def initialize_vectorstore():
    """Initialize vectorstore only when needed"""
    if st.session_state.vectorstore is None:
        try:
            with open(DEFAULT_TEXT_PATH, 'r') as file:
                text = file.read()
                text_splitter = CharacterTextSplitter(
                    separator="\n",
                    chunk_size=500,
                    chunk_overlap=100,
                    length_function=len
                )
                chunks = text_splitter.split_text(text)
                
                embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
                st.session_state.vectorstore = FAISS.from_texts(texts=chunks, embedding=embeddings)
        except FileNotFoundError:
            st.error(f"Default file {DEFAULT_TEXT_PATH} not found!")
            st.stop()

def generate_response(query):
    try:
        # Initialize vectorstore only on first use
        if st.session_state.vectorstore is None:
            initialize_vectorstore()
        
        if st.session_state.conversation_chain is None:
            memory = ConversationBufferMemory(
                memory_key='chat_history',
                return_messages=True
            )
            
            llm = ChatOpenAI(
                model_name="gpt-3.5-turbo-16k",
                temperature=0.7,
                openai_api_key=openai_api_key
            )
            
            retriever = st.session_state.vectorstore.as_retriever(
                search_kwargs={"k": 3}
            )
            
            st.session_state.conversation_chain = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=retriever,
                memory=memory
            )
        
        response = st.session_state.conversation_chain({"question": query})
        return response['answer']
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        return "I apologize, but I encountered an error processing your request. Please try again."

def check_timeout():
    """Check if 5 minutes have passed since last activity"""
    if st.session_state.last_activity:
        inactive_time = time.time() - st.session_state.last_activity
        if inactive_time > 300:  # 300 seconds = 5 minutes
            # Reset everything
            st.session_state.conversation_chain = None
            st.session_state.vectorstore = None
            st.session_state.messages = []
            st.session_state.last_activity = None
            return True
    return False

# Main chat interface
with st.form("my_form"):
    text = st.text_area('Enter your question:', '')
    submitted = st.form_submit_button("Submit")
    
    if submitted:
        # Check for timeout before processing
        if check_timeout():
            st.warning("Session timed out due to inactivity. Please start a new conversation.")
            st.rerun()
        
        response = generate_response(text)
        st.session_state.messages.append({"role": "user", "content": text})
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.session_state.last_activity = time.time()  # Update last activity time

# Display chat history
for message in st.session_state.messages:
    if message["role"] == "user":
        st.write("You:", message["content"])
    else:
        st.write("Assistant:", message["content"])

if st.button("Clear Chat"):
    st.session_state.conversation_chain = None
    st.session_state.messages = []

