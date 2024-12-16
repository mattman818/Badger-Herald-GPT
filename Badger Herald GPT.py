#Badger Herald GPT

import streamlit as st
from langchain_openai import ChatOpenAI, OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import OpenAIEmbeddings

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
    # Process the text file once when the app starts
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

# Main chat interface
with st.form("my_form"):
    text = st.text_area('Enter your question:', '')
    submitted = st.form_submit_button("Submit")
    
    if submitted:
        response = generate_response(text)
        st.session_state.messages.append({"role": "user", "content": text})
        st.session_state.messages.append({"role": "assistant", "content": response})

# Display chat history
for message in st.session_state.messages:
    if message["role"] == "user":
        st.write("You:", message["content"])
    else:
        st.write("Assistant:", message["content"])

if st.button("Clear Chat"):
    st.session_state.conversation_chain = None
    st.session_state.messages = []


