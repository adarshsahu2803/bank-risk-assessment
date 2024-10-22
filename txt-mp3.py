import boto3
from pydub import AudioSegment
import re
import os

def split_conversation(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
    
    lines = content.split('\n')
    conversation = []
    current_speaker = None
    
    for line in lines:
        if ':' in line:
            speaker, dialogue = line.split(':', 1)
            speaker = speaker.strip()
            dialogue = dialogue.strip()
            conversation.append((speaker, dialogue))
            current_speaker = speaker
        elif current_speaker and line.strip():
            conversation[-1] = (current_speaker, conversation[-1][1] + ' ' + line.strip())
    
    return conversation

def synthesize_speech(text, voice_id, output_filename):
    polly_client = boto3.client('polly')
    
    response = polly_client.synthesize_speech(
        Text=text,
        OutputFormat='mp3',
        VoiceId=voice_id
    )
    
    with open(output_filename, 'wb') as file:
        file.write(response['AudioStream'].read())

# Read and split the conversation
conversation = split_conversation('credit-risk-trancript.txt')

# Define voices for each speaker (you can change these)
voices = {
    'Teller': 'Joanna',
    'Consumer': 'Matthew'
}

# Synthesize speech for each dialogue line
audio_segments = []
for speaker, dialogue in conversation:
    temp_filename = f"temp_{speaker.lower()}_{len(audio_segments)}.mp3"
    voice_id = voices.get(speaker, 'Joanna')  # Default to Joanna if speaker not found
    
    synthesize_speech(dialogue, voice_id, temp_filename)
    audio_segment = AudioSegment.from_mp3(temp_filename)
    audio_segments.append(audio_segment)
    
    # Add a short pause between speakers
    pause = AudioSegment.silent(duration=500)  # 500 ms pause
    audio_segments.append(pause)

# Concatenate all audio segments
full_conversation = sum(audio_segments)

# Export the full conversation
full_conversation.export("credit-risk.mp3", format="mp3")

# Clean up temporary files
for filename in os.listdir():
    if filename.startswith("temp_") and filename.endswith(".mp3"):
        os.remove(filename)

print("Conversation audio created: full_conversation.mp3")