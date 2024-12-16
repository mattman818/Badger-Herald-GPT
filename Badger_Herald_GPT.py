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

# Set up the web app title
st.title("The Badger Herald GPT")

# Get the OpenAI API key from Streamlit secrets
openai_api_key = st.secrets["OPENAI_API_KEY"]

# Define the path to the default text file
DEFAULT_TEXT_PATH = "default.txt"

# Initialize session state variables to persist data between reruns
if "conversation_chain" not in st.session_state:
    st.session_state.conversation_chain = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vectorstore" not in st.session_state:
    # Process the text file and create embeddings only once when the app starts
    try:
        # Read the default text file
        with open(DEFAULT_TEXT_PATH, 'r') as file:
            text = file.read()
            # Configure text splitting parameters
            text_splitter = CharacterTextSplitter(
                separator="\n",
                chunk_size=500,  # Size of each text chunk
                chunk_overlap=100,  # Overlap between chunks to maintain context
                length_function=len
            )
            # Split the text into chunks
            chunks = text_splitter.split_text(text)
            
            # Create embeddings for the text chunks
            embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
            st.session_state.vectorstore = FAISS.from_texts(texts=chunks, embedding=embeddings)
    except FileNotFoundError:
        st.error(f"Default file {DEFAULT_TEXT_PATH} not found!")
        st.stop()

def generate_response(query):
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
        
        # Create the conversation chain
        st.session_state.conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=memory
        )
    
    # Generate response using the conversation chain
    response = st.session_state.conversation_chain({"question": query})
    return response['answer']

# Create the chat interface using a form
with st.form("my_form"):
    text = st.text_area('Enter your question:', '')
    submitted = st.form_submit_button("Submit")
    
    if submitted:
        # Generate response and update chat history
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


