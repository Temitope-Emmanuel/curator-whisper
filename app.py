import whisper
import os
import base64
from io import BytesIO
import requests
import json

# Init is ran on server startup
# Load your model to GPU as a global variable here using the variable name "model"


def init():
    global model
    # medium, large-v1, large-v2
    model_name = "large-v1"
    model = whisper.load_model(model_name)

# Inference is ran for every server call
# Reference your preloaded global model variable here.

def inference(model_inputs: dict) -> dict:
    global model

    # Parse out your arguments
    mp3BytesString = model_inputs.get('mp3BytesString', None)
    original_start_time = model_inputs.get('start_time', None)
    original_end_time = model_inputs.get('end_time', None)
    media_id = model_inputs.get('media_id',None)
    if mp3BytesString == None:
        return {'message': 'No input provided'}

    mp3Bytes = BytesIO(base64.b64decode(mp3BytesString.encode("ISO-8859-1")))
    with open('input.mp3', 'wb') as file:
        file.write(mp3Bytes.getbuffer())

    # Run the model
    result = model.transcribe("input.mp3", language="English")
    os.remove("input.mp3")
    # Return the results as a dictionary

    segments = result['segments']
    filtered_segments = []
    for data in segments:
        filtered_segments.append(
            {
                'id': data['id'],
                'seek': data['seek'],
                'start': data['start'],
                'end': data['end'],
                'text': data['text']
            },
        )

    data = {
        'text': result['text'],
        'media_id': media_id,
        'original_start_time': original_start_time,
        'original_end_time': original_end_time,
        'segments':  json.dumps(filtered_segments)
    }
    print('this is the data', data)
    requests.post(
        'https://us-central1-curator-a7ae1.cloudfunctions.net/addTranscriptToAudio', data=data)

    return data
