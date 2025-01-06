#Badger Herald GPT

# Import necessary libraries
import streamlit as st  # For creating the web interface
from langchain_openai import ChatOpenAI, OpenAI  # OpenAI's chat models
from langchain.text_splitter import CharacterTextSplitter  # For splitting text into chunks
from langchain_community.document_loaders import TextLoader  # For loading text documents
from langchain.memory import ConversationBufferMemory  # For maintaining conversation history
from langchain_community.vectorstores import FAISS  # For vector storage and similarity search
from langchain.chains import ConversationalRetrievalChain  # For creating conversation chains
from langchain_openai import OpenAIEmbeddings  # For creating text embeddings
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)


# Set up the web app title
st.title("The Badger GPT")

# Get the OpenAI API key from Streamlit secrets
openai_api_key = st.secrets["OPENAI_API_KEY"]

# Define the path to the default text file
DEFAULT_TEXT_PATH = "default.txt"

# Initialize session state
if "conversation_chain" not in st.session_state:
    st.session_state.conversation_chain = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

def initialize_vectorstore():
    """Initialize vectorstore only when needed"""
    if st.session_state.vectorstore is None:
        with st.spinner("Loading knowledge base..."):  # Add loading indicator
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
    # Initialize vectorstore only when generating a response
    initialize_vectorstore()
    
    # Initialize the conversation chain if it doesn't exist
    if st.session_state.conversation_chain is None:
        # Set up conversation memory
        memory = ConversationBufferMemory(
            memory_key='chat_history',
            return_messages=True
        )
        
        # Initialize the language model
        llm = ChatOpenAI(
            model_name="gpt-3.5-turbo-16k",
            temperature=0.7,  # Controls randomness in responses
            openai_api_key=openai_api_key
        )
        
        # Set up the retriever with search parameters
        retriever = st.session_state.vectorstore.as_retriever(
            search_kwargs={"k": 3}  # Number of relevant chunks to retrieve
        )
        
        # Create the conversation chain with system message
        st.session_state.conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=memory,
            chain_type="stuff",
            return_source_documents=True,
            combine_docs_chain_kwargs={
                "prompt": ChatPromptTemplate.from_messages([
                    SystemMessagePromptTemplate.from_template(
                        "If you don't have sufficient information, attempt to provide related insights instead of saying you can't answer. \n\nContext: {context}"
                    ),
                    HumanMessagePromptTemplate.from_template("{question}")
                ])
            })

    
    # Generate response using the conversation chain
    response = st.session_state.conversation_chain({"question": query})
    return response['answer']

# Add sample questions to the sidebar
st.sidebar.header("Sample Questions")
sample_questions = [
    "How is the UW madison womens volleyball team doing?",
    "How is the badger basketball team doing?",
    "Where are some good bookstores on campus?",
    "Are there any art exhibitions on campus?",
    "Is there any theatre to watch on campus?",
    "Are the northern lights visible from UW madison?"
]

st.sidebar.write("Try asking these questions:")
for question in sample_questions:
    if st.sidebar.button(question):
        initialize_vectorstore()  # Only initialize when button is clicked
        response = generate_response(question)
        st.session_state.messages.append({"role": "user", "content": question})
        st.session_state.messages.append({"role": "assistant", "content": response})

# Create the chat interface using a form
with st.form("my_form"):
    text = st.text_area('Enter your question:', '')
    submitted = st.form_submit_button("Submit")
    
    if submitted and text:  # Add text check to prevent empty submissions
        initialize_vectorstore()  # Only initialize when form is submitted
        response = generate_response(text)
        st.session_state.messages.append({"role": "user", "content": text})
        st.session_state.messages.append({"role": "assistant", "content": response})

# Display the chat history
for message in st.session_state.messages:
    if message["role"] == "user":
        st.write("You:", message["content"])
    else:
        st.write("Assistant:", message["content"])

# Add a button to clear the chat history
if st.button("Clear Chat"):
    st.session_state.conversation_chain = None
    st.session_state.messages = []
