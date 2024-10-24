import streamlit as st
import pandas as pd
import io
import tempfile
import os
import assemblyai as aai
import json
import boto3
from collections import deque
from pydub import AudioSegment
from transformers import pipeline

st.set_page_config(layout="wide",)

access_key = os.getenv('ACCESS_KEY_ID')
secret_access_key = os.getenv('SECRET_ACCESS_KEY')
region_name = os.getenv('DEFAULT_REGION')

bedrock_runtime = boto3.client(service_name='bedrock-runtime', aws_access_key_id=access_key,aws_secret_access_key=secret_access_key, region_name=region_name)

if 'dataframe' not in st.session_state:
    st.session_state['dataframe'] = pd.DataFrame(columns=[
        'Category', 'Issue/Risk', 'Risk Description', 'Potential Impact [1-5]',
        'Likelihood [1-5]', 'Risk Ranking', 'Primary Point of Contact',
        'Description of Monitoring', 'Comment'
    ])

st.session_state['counter'] = 0

# Function to transcribe audio
def transcribe_audio(audio_file):
    aai.settings.api_key = "1ced3445571e4a07a6a7535ae94a405d"
    transcriber = aai.Transcriber()

    transcript = transcriber.transcribe(audio_file)

    return transcript.text

# Function to extract Q&A from transcript
def extract_qa(transcript):
    prompt = (
        "You are a trade compliance analyst. Based on the given conversation transcript, generate 7 important question-answer pairs."
        "Each pair should focus on key information, motives for trade execution, or significant points from the conversation."
        "Format your response as follows:"
        "Q1: [Question 1]\n\n"
        "A1: [Answer 1]\n"
        "\n"
        "Q2: [Question 2]\n\n"
        "A2: [Answer 2]\n"
        "\n"
        "Q3: [Question 3]\n\n"
        "A3: [Answer 3]\n"
        "\n"
        "... and so on."
        "Ensure that the questions and answers are directly based on the content of the conversation, without drawing external conclusions."
        "Also make sure these que-ans pairs are informative and professional. Directly start your response with Q1:..."
        "Here is the conversation text:\n\n"
        f"{transcript}\n\n"
        "Question-Answer Pairs:"
    )
    request_body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 1000,
        "temperature": 0.7,
        "messages": [
            {"role": "user", "content": prompt}
        ]
    })
    response = bedrock_runtime.invoke_model(
        modelId='anthropic.claude-3-5-sonnet-20240620-v1:0',
        body=request_body,
    )
    response_body = json.loads(response['body'].read())
    generated_text = response_body['content'][0]['text']
    
    return generated_text

def generate_df_entry(transcript):
    prompt = (
        "You are a trade compliance analyst. Based on the given conversation transcript, generate a dictionary that can be used as a dataframe entry for the following dataframe structure:"
         """
        df = pd.DataFrame(columns=[
            'Category', 'Issue/Risk', 'Risk Description', 'Potential Impact [1-5]',
            'Likelihood [1-5]', 'Risk Ranking', 'Primary Point of Contact',
            'Description of Monitoring', 'Comment'
        ])
        """
        "Ensure the entry is professional and follows this exact format:"
        """
        {
            'Category': '',
            'Issue/Risk': '',
            'Risk Description': '',
            'Potential Impact [1-5]': 0,
            'Likelihood [1-5]': 0,
            'Risk Ranking': '',
            'Primary Point of Contact': '',
            'Description of Monitoring': '',
            'Comment': ''
        }
        """
         "Your response should be only this dictionary, without any additional text or explanations."
        "Use integer values for 'Potential Impact [1-5]' and 'Likelihood [1-5]'."
        "Here is the conversation transcript to analyze:\n\n"
        f"{transcript}\n\n"
    )

    request_body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 1000,
        "temperature": 0.7,
        "messages": [
            {"role": "user", "content": prompt}
        ]
    })

    response = bedrock_runtime.invoke_model(
        modelId='anthropic.claude-3-5-sonnet-20240620-v1:0',
        body=request_body,
    )
    response_body = json.loads(response['body'].read())
    generated_text = response_body['content'][0]['text']

    return generated_text

# Streamlit app
st.title("Bank Risk Assessment App")

# if 'dataframe' in st.session_state:
#     st.write(st.session_state['dataframe'])

# Create two columns
col1, colm, col2 = st.columns([2,0.2,2])

with col1:
    st.header("Upload Voice File")
    uploaded_file = st.file_uploader("Choose an audio file", type=['wav', 'mp3', 'ogg'])
    
    if uploaded_file is not None:
        # Save the uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        
        # Transcribe the audio
        transcript = transcribe_audio(tmp_file_path)
        st.subheader("Transcript")
        st.write(transcript)
        
        # Remove the temporary file
        os.unlink(tmp_file_path)

with col2:
    st.header("Important Q&A")
    if uploaded_file is not None:
        qa_pair = extract_qa(transcript)
        splits = qa_pair.split('\n')
        for message in splits:
            if message.startswith("Q"):
                st.chat_message("ai").write(message[3:])
            elif message.startswith("A"):
                st.chat_message("human").write(message[3:])


if uploaded_file is not None:
    entry = generate_df_entry(transcript)
    new_entry = eval(entry)  # Convert the string response to a dictionary
    new_df = pd.concat([st.session_state['dataframe'], pd.DataFrame([new_entry])], ignore_index=True)        
    st.write(pd.DataFrame(new_df))