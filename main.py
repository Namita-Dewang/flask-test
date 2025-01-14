from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from dotenv import load_dotenv
import logging
from smallest import Smallest
from openai import OpenAI
import time
import torch
from faster_whisper import WhisperModel

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})  # Configure CORS for API endpoints
UPLOAD_FOLDER = 'audio_files'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize clients
smallest_api_key = os.getenv("SMALLEST_API_KEY")
smallest_client = Smallest(api_key=smallest_api_key)

deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
deepseek_client = OpenAI(api_key=deepseek_api_key, base_url="https://api.deepseek.com")

# Initialize Whisper model
whisper_model = WhisperModel("small", device="cpu")
logger.info(f"Whisper model loaded on: cpu")

def transcribe_audio(audio_file):
    try:
        temp_path = os.path.join(UPLOAD_FOLDER, "temp_audio.wav")
        audio_file.save(temp_path)
        segments, info = whisper_model.transcribe(temp_path)
        transcription = " ".join([segment.text for segment in segments])
        os.remove(temp_path)
        return {"success": True, "transcription": transcription}
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        return {"success": False, "error": str(e)}

def send_to_deepseek(transcription):
    try:
        response = deepseek_client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": transcription},
            ],
            stream=False
        )
        return {"success": True, "response": response.choices[0].message.content}
    except Exception as e:
        logger.error(f"DeepSeek API error: {e}")
        return {"success": False, "error": str(e)}

# API Endpoints
@app.route('/api/transcribe', methods=['POST'])
def api_transcribe():
    try:
        if 'audio' not in request.files:
            return jsonify({"success": False, "error": "No audio file provided"}), 400
            
        audio_file = request.files['audio']
        result = transcribe_audio(audio_file)
        
        if not result["success"]:
            return jsonify(result), 500

        return jsonify({
            "success": True,
            "transcription": result["transcription"]
        })

    except Exception as e:
        logger.error(f"Transcription API error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/chat', methods=['POST'])
def api_chat():
    try:
        if not request.json or 'message' not in request.json:
            return jsonify({"success": False, "error": "No message provided"}), 400

        result = send_to_deepseek(request.json['message'])
        
        if not result["success"]:
            return jsonify(result), 500

        return jsonify({
            "success": True,
            "response": result["response"]
        })

    except Exception as e:
        logger.error(f"Chat API error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/process-audio', methods=['POST'])
def api_process_audio():
    try:
        if 'audio' not in request.files:
            return jsonify({"success": False, "error": "No audio file provided"}), 400
            
        audio_file = request.files['audio']
        
        # Get transcription
        trans_result = transcribe_audio(audio_file)
        if not trans_result["success"]:
            return jsonify(trans_result), 500

        # Get DeepSeek response
        chat_result = send_to_deepseek(trans_result["transcription"])
        if not chat_result["success"]:
            return jsonify(chat_result), 500

        # Generate audio response
        filename = f"response_{int(time.time())}.wav"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        smallest_client.synthesize(chat_result["response"], save_as=file_path)
        
        return jsonify({
            "success": True,
            "transcription": trans_result["transcription"],
            "chat_response": chat_result["response"],
            "audio_url": f"/api/audio/{filename}"
        })

    except Exception as e:
        logger.error(f"Process audio API error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/audio/<filename>')
def api_serve_audio(filename):
    try:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if not os.path.exists(file_path):
            return jsonify({"success": False, "error": "Audio file not found"}), 404
        
        return send_file(
            file_path,
            mimetype='audio/wav',
            as_attachment=False
        )
    except Exception as e:
        logger.error(f"Audio serve API error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/health', methods=['GET'])
def api_health_check():
    return jsonify({
        "success": True,
        "status": "healthy",
        "whisper_device": "cuda" if torch.cuda.is_available() else "cpu"
    })

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
