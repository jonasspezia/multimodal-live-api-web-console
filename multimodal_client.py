import os
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from google import genai
import asyncio

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)

# Initialize the Gemini client
client = genai.Client(
    api_key=os.getenv("REACT_APP_GEMINI_API_KEY"),
    http_options={'api_version': 'v1alpha'}
)

# Model configuration
MODEL_ID = "gemini-2.0-flash-exp"
CONFIG = {
    "response_modalities": ["TEXT"],
    "generation_config": {
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 40,
        "max_output_tokens": 1024,
    },
    "speech_config": {
        "voice_config": {
            "prebuilt_voice_config": {
                "voice_name": "Aoede"
            }
        }
    }
}

async def process_message(message):
    try:
        async with client.aio.live.connect(model=MODEL_ID, config=CONFIG) as session:
            # Send message
            await session.send(message, end_of_turn=True)
            
            # Collect responses
            response_text = ""
            async for response in session.receive():
                if response.text is not None:
                    response_text += response.text
            
            return response_text
    except Exception as e:
        return str(e)

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    message = data.get('message', '')
    
    if not message:
        return jsonify({'error': 'No message provided'}), 400
    
    # Run async code in sync context
    response = asyncio.run(process_message(message))
    return jsonify({'response': response})

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    port = int(os.getenv('PORT', 8080))
    app.run(host='0.0.0.0', port=port)
