import chromadb
import os
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import pipeline
import torch
import base64
import textwrap
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from constants import CHROMA_SETTINGS
from gtts import gTTS
import platform
import base64

# model and tokenizer loading
tokenizer = AutoTokenizer.from_pretrained("MBZUAI/LaMini-T5-738M")
model = AutoModelForSeq2SeqLM.from_pretrained("MBZUAI/LaMini-T5-738M")

@st.cache_resource
def llm_pipeline():
    pipe = pipeline(
        'text2text-generation',
        model=model,
        tokenizer=tokenizer,
        max_length=312,
        do_sample=True,
        temperature=0.3,
        top_p=0.95
    )
    local_llm = HuggingFacePipeline(pipeline=pipe)
    return local_llm

@st.cache_resource
def qa_llm():
    llm = llm_pipeline()
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    client = chromadb.PersistentClient(path="db_metadata_v5")
    vector_db = Chroma(client=client, embedding_function=embeddings)
    retriever = vector_db.as_retriever()
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)
    return qa

def process_answer(instruction):
    response = ''
    instruction = instruction
    qa = qa_llm()
    generated_text = qa(instruction)
    answer = generated_text['result']
    st.session_state.answer = answer  # Store answer in session_state
    return answer, generated_text

def speak_text(text):
    # Save the generated text to an audio file
    tts = gTTS(text=text, lang='en')
    audio_file_path = "generated_audio.mp3"
    tts.save(audio_file_path)
    
    # Play the audio file
    play_audio(audio_file_path)

    # os.remove(audio_file_path)

def play_audio(audio_file_path):
    # Check the platform to determine the appropriate command for playing audio
    if platform.system() == "Darwin":
        os.system("afplay " + audio_file_path)
    elif platform.system() == "Linux":
        os.system("aplay " + audio_file_path)
    elif platform.system() == "Windows":
        os.system("start " + audio_file_path)
    else:
        st.warning("Unsupported platform for text-to-speech")
st.title("Search Your PDF 🔍📄")

with st.sidebar:
    st.markdown("<h1 style='text-align:center;font-family:Georgia;font-size:26px;'>🧑‍⚖️ Search-PDF </h1>", unsafe_allow_html=True)
    st.markdown("<h7 style='text-align:left;font-size:20px;'>This app is a smart platform that is integrated into an easy-to-use platform."
            "This gives instant access to  information of document contents and remove the need for laborious manual research in books</h7>", unsafe_allow_html=True)
    st.markdown("-------")
    st.markdown("<h2 style='text-align:center;font-family:Georgia;font-size:20px;'>Features</h1>", unsafe_allow_html=True)
    
    st.markdown(" - Text to speech direct implementation")
    st.markdown(" - ")
    st.markdown("-------")
    

question = st.text_area("Enter your Question")
if st.button("Ask"):
    st.info("Your Question: " + question)
    st.info("Your Answer")
    answer, metadata = process_answer(question)
    st.write(answer)
    st.write(metadata)

    #if st.button("Listen to Answer"):
    speak_text(answer)
