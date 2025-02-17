import streamlit as st
import chromadb
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import torch
import tempfile
import time
import gc
import os
## Read from env
# from dotenv import load_dotenv
# load_dotenv()
# secret_token = os.getenv('token')

## Read From config
# from config import CHROMA_DB_PATH, 


def generate_response(user_input, system_prompt=None):
    """Generate response using the fine-tuned model and measure response time."""
    start_time = time.time()
    
    # Default system prompt for general questions
    if system_prompt is None:
        system_prompt = f"""
        You are a highly knowledgeable AI assistant. Your task is to answer the user's question based on the context retrieved from the knowledge base.
        
        **Instructions:**
        1. Do not include any information not present in the context.
        2. Provide a concise and accurate answer.
        """
    
    # Format the conversation using the chat template
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input},
    ]
    # Tokenize the input
    inputs = tokenizer.apply_chat_template(messages, tokenize=False)
    inputs = tokenizer(inputs, return_tensors="pt", padding=True, truncation=True).input_ids.to("cuda")

    # Generate response
    with torch.no_grad():
        outputs = model.generate(inputs, max_new_tokens=100, temperature=0.2, top_p=0.8, do_sample=True)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Measure response time
    if not system_prompt:
        # response = response.replace(system_prompt, '')
        # response = response.replace(user_input, '')
        response = ",".join(response.split('AI:')[1:])
    response_time = round(time.time() - start_time, 2)
    print(f"Generated response: {response}")
    
    # Clear CUDA cache and garbage collect
    torch.cuda.empty_cache()
    gc.collect()
    return response, response_time

# Initialize ChromaDB
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="rag_knowledge")
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def query_rag(prompt):
    """Retrieve answer from indexed documents using RAG."""
    try:
        # Retrieve relevant documents from ChromaDB
        results = collection.query(query_texts=[prompt], n_results=2)
        if not results["documents"]:
            return "No relevant documents found in the knowledge base.", 0.0
        
        context = "\n".join(results["documents"][0])
        
        # Custom system prompt for RAG
        rag_system_prompt = f"""
        You are a highly knowledgeable AI assistant. Your task is to answer the user's question based on the context retrieved from the knowledge base.
        
        **Instructions:**
        1. Answer user query from the below context.
        **Retrieved Context:**
        {context}
        1. Answer the user query using only the information provided in the context above.
        2. Do not include any information not present in the context.
        3. Only if there is no infomration available in the context, provide general answer.
        4. Provide a concise and accurate answer.
        """
        # Generate response using the RAG system prompt
        response, response_time = generate_response(prompt, system_prompt=rag_system_prompt)
        response = response.replace(prompt, '')
        response = response.replace(rag_system_prompt, '')
        response = ",".join(response.split('AI:')[1:])
        return response, response_time
    except Exception as e:
        return f"Error querying RAG: {str(e)}", 0.0
# Example usage in Streamlit
def extract_main_content(url):
    """Extract the full page source and content from <p> tags using Selenium."""
    try:
        # Set up Selenium WebDriver
        options = webdriver.ChromeOptions()
        options.add_argument("--headless")  # Run in headless mode
        
        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
        
        # Load the webpage
        driver.get(url)
        
        # Wait for the page to load (adjust the sleep time as needed)
        time.sleep(5)
        
        # Get the full page source
        page_source = driver.page_source
        
        # Parse the page source with BeautifulSoup
        soup = BeautifulSoup(page_source, "html.parser")
        
        # Extract content from all <p> tags using BeautifulSoup
        paragraphs = soup.find_all("p")  # Find all <p> tags
        content = " ".join([p.get_text() for p in paragraphs if p.get_text().strip()])  # Join non-empty paragraphs
        
        # Close the browser
        driver.quit()
        
        # Return the content as a list of documents
        return [Document(page_content=content)]
    except Exception as e:
        st.sidebar.error(f"Error extracting content: {str(e)}")
        return []

def add_url_to_index(url):
    """Extract content from URL using Selenium and add to knowledge base."""
    try:
        # Use Selenium to extract content
        docs = extract_main_content(url)
        if not docs:
            st.sidebar.error("No content extracted from the URL.")
            return
        
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        chunks = text_splitter.split_documents(docs)
        texts = [chunk.page_content for chunk in chunks]
        
        # Generate embeddings for the chunks
        embeddings = embedding_model.embed_documents(texts)
        
        # Add documents and embeddings to the ChromaDB collection
        for i, text in enumerate(texts):
            if len(embeddings[i]) > 0:  # Ensure embeddings are not empty
                collection.add(
                    documents=[text],
                    embeddings=[embeddings[i]],
                    ids=[f"url_{i}"]  # Unique ID for each document
                )
        
        st.sidebar.success("Content added to knowledge base!")
    except Exception as e:
        st.sidebar.error(f"Error adding URL to index: {str(e)}")

def add_pdf_to_index(uploaded_file):
    """Extract text from PDF and add to knowledge base."""
    try:
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(uploaded_file.getvalue())
            temp_path = temp_file.name
        loader = PyMuPDFLoader(temp_path)
        docs = loader.load()
        os.remove(temp_path)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        chunks = text_splitter.split_documents(docs)
        texts = [chunk.page_content for chunk in chunks]
        embeddings = embedding_model.embed_documents(texts)
        for i, text in enumerate(texts):
            collection.add(documents=[text], embeddings=[embeddings[i]], ids=[f"pdf_{i}"])
        st.sidebar.success("PDF content added to knowledge base!")
    except Exception as e:
        st.sidebar.error(f"Error adding PDF to index: {str(e)}")

# Streamlit UI
st.set_page_config(page_title="Chatbot", layout="wide")

# Sidebar: Settings
st.sidebar.title("Settings")

# Model Selection
st.sidebar.subheader("Select Model")
model_option = st.sidebar.radio("Choose a Model", ["Custom Model", "SmolLLM"])
MODEL_NAME = "fine_tuned_llm" if model_option == "Custom Model" else "HuggingFaceTB/SmolLM2-1.7B-Instruct"
# Load the model and tokenizer
# MODEL_NAME = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype="auto", device_map="cuda")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Define a chat template for the tokenizer
tokenizer.chat_template = """
{% for message in messages %}
    {% if message['role'] == 'system' %}
        {{ message['content'] }}
    {% elif message['role'] == 'user' %}
        ### User: {{ message['content'] }}
    {% elif message['role'] == 'assistant' %}
        ### Assistant: {{ message['content'] }}
    {% endif %}
{% endfor %}
"""

st.sidebar.subheader("Search Mode")
mode = st.sidebar.radio("Select Mode", ["General", "Document Search"])

# Sidebar: Add Knowledge
st.sidebar.subheader("Add to Knowledge")
url_input = st.sidebar.text_input("Enter URL")
if st.sidebar.button("Submit URL"):
    if url_input:
        add_url_to_index(url_input)
    else:
        st.sidebar.error("Please enter a valid URL")

uploaded_file = st.sidebar.file_uploader("Upload PDF File", type=["pdf"])
if st.sidebar.button("Submit PDF"):
    if uploaded_file:
        add_pdf_to_index(uploaded_file)
    else:
        st.sidebar.error("Please upload a PDF file")

# Main Section: Chat
st.title("The Law Reporters's AI Assistant")
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.chat_input("Type your message...")
if user_input:
    if mode == "General":
        response, response_time = generate_response(user_input)
    else:
        response, response_time = query_rag(user_input)
    
    # Append user message with is_bot=False
    st.session_state.chat_history.append(("You", user_input, False))
    # Append bot message with is_bot=True
    st.session_state.chat_history.append(("Bot", f"{response} <span style='font-size: small; color: gray;'>({response_time}s)</span>", True))

# Display Chat History
for speaker, msg, is_bot in st.session_state.chat_history:
    if is_bot:
        st.markdown(f"**{speaker}:** {msg}", unsafe_allow_html=True)
    else:
        st.write(f"**{speaker}:** {msg}")