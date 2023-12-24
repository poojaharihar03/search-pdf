import chromadb
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

# model and tokenizer loading
tokenizer = AutoTokenizer.from_pretrained("MBZUAI/LaMini-T5-738M")
model = AutoModelForSeq2SeqLM.from_pretrained("MBZUAI/LaMini-T5-738M")

@st.cache_resource
def llm_pipeline():
    pipe = pipeline(
        'text2text-generation',
        model=model,
        tokenizer=tokenizer,
        max_length=max_length,
        do_sample=True,
        temperature=temp,
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


st.title("Search Your PDF 🔍📄")

with st.sidebar:
    st.markdown("<h1 style='text-align:center;font-family:Georgia;font-size:26px;'>🧑‍⚖️ Search-PDF </h1>", unsafe_allow_html=True)
    st.markdown("<h7 style='text-align:left;font-size:20px;'>This app is a smart platform that is integrated into an easy-to-use platform."
            "This gives instant access to  information of document contents and remove the need for laborious manual research in books</h7>", unsafe_allow_html=True)
    st.markdown("-------")
    st.markdown("<h2 style='text-align:center;font-family:Georgia;font-size:20px;'>Features</h1>", unsafe_allow_html=True)
    
    st.markdown(" - Users can adjust token length to control the length of generated responses, allowing for customization based on specific requirements or constraints.")
    st.markdown(" - Users can adjust the temp to control response randomness. Higher values (e.g., 0.5) produce diverse but less focused responses, while low values (e.g., 0.1) result in more focused but less varied answers.")
    st.markdown("-------")
    st.markdown("<h2 style='text-align:center;font-family:Georgia;font-size:20px;'>Advanced Features</h1>", unsafe_allow_html=True)
    max_length=st.slider("Token Max Length", min_value=128, max_value=1024, value=128, step=128)
    temp=st.slider("Temperature", min_value=0.1, max_value=1.0, value=0.1, step=0.1)

question = st.text_area("Enter your Question")
if st.button("Ask"):
    st.info("Your Question: " + question)
    st.info("Your Answer")
    answer, metadata = process_answer(question)
    st.write(answer)
    st.write(metadata)























# import chromadb
# import streamlit as st 
# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
# from transformers import pipeline
# import torch
# import base64
# import textwrap
# from langchain.embeddings import SentenceTransformerEmbeddings
# from langchain.vectorstores import Chroma 
# from langchain.llms import HuggingFacePipeline
# from langchain.chains import RetrievalQA
# from constants import CHROMA_SETTINGS
# from gtts import gTTS
# from io import BytesIO

# # model and tokenizer loading
# tokenizer = AutoTokenizer.from_pretrained("MBZUAI/LaMini-T5-738M")
# model = AutoModelForSeq2SeqLM.from_pretrained("MBZUAI/LaMini-T5-738M")

# @st.cache_resource
# def llm_pipeline():
#     pipe = pipeline(
#         'text2text-generation',
#         model=model,
#         tokenizer=tokenizer,
#         max_length=max_length,
#         do_sample=True,
#         temperature=temp,
#         top_p=0.95
#     )
#     local_llm = HuggingFacePipeline(pipeline=pipe)
#     return local_llm

# @st.cache_resource
# def qa_llm():
#     llm = llm_pipeline()
#     embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
#     client = chromadb.PersistentClient(path="db_metadata_v5")
#     vector_db = Chroma(client=client, embedding_function=embeddings)
#     retriever = vector_db.as_retriever()
#     qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=False)
#     return qa

# def process_answer(instruction):
#     response = ''
#     instruction = instruction
#     qa = qa_llm()
#     generated_text = qa(instruction)
#     answer = generated_text['result']
#     return answer, generated_text
# st.title("Search Your PDF 🔍📄")
# with st.sidebar:
#     st.markdown("<h1 style='text-align:center;font-family:Georgia;font-size:26px;'>🧑‍⚖️ Search-PDF </h1>", unsafe_allow_html=True)
#     st.markdown("<h7 style='text-align:left;font-size:20px;'>This app is a smart platform that is integrated into an easy-to-use platform."
#             "This gives instant access to  information of document contents and remove the need for laborious manual research in books</h7>", unsafe_allow_html=True)
#     st.markdown("-------")
#     st.markdown("<h2 style='text-align:center;font-family:Georgia;font-size:20px;'>Features</h1>", unsafe_allow_html=True)
    
#     st.markdown(" - Users can adjust token length to control the length of generated responses, allowing for customization based on specific requirements or constraints.")
#     st.markdown(" - Users can adjust the temp to control response randomness. Higher values (e.g., 0.5) produce diverse but less focused responses, while low values (e.g., 0.1) result in more focused but less varied answers.")
#     st.markdown("-------")
#     st.markdown("<h2 style='text-align:center;font-family:Georgia;font-size:20px;'>Advanced Features</h1>", unsafe_allow_html=True)
#     max_length=st.slider("Token Max Length", min_value=128, max_value=1024, value=128, step=128)
#     temp=st.slider("Temperature", min_value=0.1, max_value=1.0, value=0.1, step=0.1)
#     input_option = st.radio("Choose input type:", ["Text", "Voice"])

# # st.title("Search Your PDF 🔍📄")

# # # Add a sidebar with app description
# # st.sidebar.title("About the App")
# # st.sidebar.markdown(
# #     """
# #     This is a Gen-AI powered Question and Answering app that responds to questions about your PDF File.
# #     It provides information and source from where it is generated
# #     """
# # )

# # Add option for user input (text or voice)
# if input_option == "Text":
#     question = st.text_area("Enter your Question")

#     if st.button("Ask"):
#         st.info("Your Question: " + question)
#         st.info("Your Answer")
#         answer, metadata = process_answer(question)
#         st.write(answer)
#         st.write(metadata)
# elif input_option == "Voice":
#     audio_file = st.file_uploader("Upload Voice File (WAV or MP3)", type=["wav", "mp3"])

#     if audio_file is not None:
#         audio_content = audio_file.read()

#         # Process audio content and generate text
#         # (You may need to add code for converting audio to text, depending on your specific requirements)
#         # For demonstration, let's assume you have the text from the audio
#         text_from_audio = "This is a sample text generated from the audio."

#         st.info("Text from Voice: " + text_from_audio)

#         if st.button("Read Out"):
#             # Use gTTS to convert text to speech
#             tts = gTTS(text_from_audio, lang="en")
#             audio_stream = BytesIO()
#             tts.save(audio_stream)
#             st.audio(audio_stream, format="audio/mp3")
        































