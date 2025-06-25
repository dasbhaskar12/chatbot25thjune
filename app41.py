import streamlit as st
import requests
from bs4 import BeautifulSoup
import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, util
import torch
from openai import OpenAI

# Load .env variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    st.error("Missing OpenAI API key in .env file.")
    st.stop()

# Initialize OpenAI and Sentence Transformer
client = OpenAI(api_key=openai_api_key)
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Pages to crawl for sbinfowaves.com
URLS = [
    "https://sbinfowaves.com/",
    "https://sbinfowaves.com/about-us/",
    "https://sbinfowaves.com/services/",
    "https://sbinfowaves.com/portfolio/",
    "https://sbinfowaves.com/contact-us/"
]

# Extract clean text from a given URL
def extract_text(url):
    try:
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        elements = soup.find_all(['p', 'li', 'h1', 'h2', 'h3'])
        text = " ".join([el.get_text(strip=True) for el in elements])
        return text
    except Exception as e:
        return ""

# Chunk text into smaller pieces
def chunk_text(text, chunk_size=150):
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

# Build knowledge base
@st.cache_data
def build_knowledge_base():
    all_chunks = []
    for url in URLS:
        text = extract_text(url)
        chunks = chunk_text(text)
        all_chunks.extend(chunks)
    embeddings = embedder.encode(all_chunks, convert_to_tensor=True)
    return all_chunks, embeddings

# Retrieve relevant chunks using semantic similarity
def find_relevant_chunks(question, chunks, embeddings, top_k=3):
    question_embedding = embedder.encode(question, convert_to_tensor=True)
    similarities = util.pytorch_cos_sim(question_embedding, embeddings)[0]
    top_indices = torch.topk(similarities, k=top_k).indices
    return [chunks[i] for i in top_indices]

# UI
st.title("SB Infowaves Chatbot 🤖")
question = st.text_input("Ask a question about SB Infowaves:")

if question:
    with st.spinner("Searching knowledge base..."):
        chunks, embeddings = build_knowledge_base()
        relevant = find_relevant_chunks(question, chunks, embeddings)

        prompt = f"Answer the question based on the context below:\n\n{''.join(relevant)}\n\nQuestion: {question}"

        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant for SB Infowaves."},
                    {"role": "user", "content": prompt}
                ]
            )
            answer = response.choices[0].message.content
            st.markdown("### AI Response")
            st.write(answer)
        except Exception as e:
            st.error(f"OpenAI API error: {str(e)}")
