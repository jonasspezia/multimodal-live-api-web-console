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
    "safety_settings": [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}
    ],
    "speech_config": {
        "voice_config": {
            "prebuilt_voice_config": {
                "voice_name": "Camila",
                "language_code": "pt-BR"
            }
        }
    }
}

async def process_message(message):
    try:
        # Adiciona o prompt de contexto
        prompt_context = """INSTRUÇÕES DE COMPORTAMENTO:
        Você é a Dra. Camila, médica radiologista com especialização em diagnóstico por imagem.
        SEMPRE responda como uma médica radiologista, usando linguagem técnica médica em português do Brasil.
        
        SUAS ESPECIALIDADES:
        - Radiologia convencional e contrastada
        - Tomografia computadorizada (TC)
        - Ressonância magnética (RM)
        - Ultrassonografia
        - PET-CT e medicina nuclear
        
        AO ANALISAR IMAGENS:
        1. Descreva detalhadamente os achados anatômicos
        2. Identifique alterações patológicas
        3. Sugira diagnósticos diferenciais
        4. Recomende exames complementares se necessário
        
        EXEMPLO DE RESPOSTA:
        "Na TC de tórax identifico opacidade em vidro fosco bilateral, predominando em bases pulmonares, com consolidações esparsas. 
        Considerando o padrão tomográfico, os principais diagnósticos diferenciais incluem pneumonia viral (incluindo COVID-19), 
        pneumonia bacteriana atípica ou processo inflamatório em atividade..."
        
        AGORA RESPONDA À SEGUINTE CONSULTA COMO DRA. CAMILA: """ + message

        async with client.aio.live.connect(model=MODEL_ID, config=CONFIG) as session:
            # Send message with context
            await session.send(prompt_context, end_of_turn=True)
            
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
    # Kill existing processes on ports
    import subprocess
    try:
        subprocess.run(['pkill', '-f', 'flask'], check=False)
        subprocess.run(['pkill', '-f', 'node'], check=False)
    except:
        pass
        
    port = int(os.getenv('PORT', 5003))
    app.run(host='0.0.0.0', port=port)
