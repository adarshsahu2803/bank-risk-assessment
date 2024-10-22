import streamlit as st
import pandas as pd
from pydub import AudioSegment
import io
import tempfile
import os
from transformers import pipeline
import assemblyai as aai

st.set_page_config(layout="wide",)

# Initialize the DataFrame with a dummy entry
df = pd.DataFrame([
    {
        'Category': 'Cybersecurity',
        'Issue/Risk': 'Data Breach',
        'Risk Description': 'Unauthorized access to sensitive customer information',
        'Potential Impact [1-5]': 4,
        'Likelihood [1-5]': 3,
        'Risk Ranking': 'High',
        'Primary Point of Contact': 'John Doe (IT Security Manager)',
        'Description of Monitoring': 'Regular security audits and penetration testing',
        'Comment': 'Implement additional encryption and access controls'
    }
], columns=[
    'Category', 'Issue/Risk', 'Risk Description', 'Potential Impact [1-5]',
    'Likelihood [1-5]', 'Risk Ranking', 'Primary Point of Contact',
    'Description of Monitoring', 'Comment'
])

# Function to transcribe audio
def transcribe_audio(audio_file):
    aai.settings.api_key = "1ced3445571e4a07a6a7535ae94a405d"
    transcriber = aai.Transcriber()

    transcript = transcriber.transcribe(audio_file)

    return transcript.text

# Function to extract Q&A from transcript
def extract_qa(transcript):
    qa_pipeline = pipeline("question-generation")
    questions = qa_pipeline(transcript)
    qa_pairs = []
    for q in questions[:5]:  # Limit to 5 questions
        answer = pipeline("question-answering")(question=q['question'], context=transcript)
        qa_pairs.append((q['question'], answer['answer']))
    return qa_pairs

# Streamlit app
st.title("Risk Management and Voice Transcription App")

# Display the DataFrame
st.dataframe(df)

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
        qa_pairs = extract_qa(transcript)
        for i, (question, answer) in enumerate(qa_pairs, 1):
            st.subheader(f"Q{i}: {question}")
            st.write(f"A: {answer}")



# This Streamlit application does the following:

# 1. Displays a DataFrame at the top with the specified columns.
# 2. Divides the page into two columns.
# 3. In the left column:
#    - Allows the user to upload an audio file.
#    - Transcribes the uploaded audio file and displays the transcript.
# 4. In the right column:
#    - Extracts 4-5 important question-answer pairs from the transcript and displays them.

# To run this application, you'll need to install the required libraries:

# ```
# pip install streamlit pandas speechrecognition pydub transformers torch
# ```

# You may also need to install ffmpeg for audio processing:

# ```
# apt-get install ffmpeg
# ```

# To run the Streamlit app, save the code in a file (e.g., `app.py`) and run:

# ```
# streamlit run app.py
# ```

# Note: The question-answering and question-generation models used in this example are basic implementations. For a production environment, you might want to use more sophisticated models or APIs for better results.

# Also, be aware that the transcription and Q&A extraction might take some time, especially for longer audio files. You may want to add progress indicators or optimize these processes for a better user experience.
    