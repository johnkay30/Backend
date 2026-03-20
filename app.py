from flask import Flask, request, jsonify
from flask_cors import CORS
from sentence_transformers import SentenceTransformer, util
import os

app = Flask(__name__)
CORS(app) # This allows your Vercel frontend to talk to this Python backend

# Load a lightweight, high-performance model
model = SentenceTransformer('all-MiniLM-L6-v2')

@app.route('/get-score', methods=['POST'])
def get_score():
    try:
        data = request.json
        m_ans = data.get('model_answer', '')
        s_ans = data.get('student_answer', '')

        # AI Magic: Calculate semantic similarity
        emb1 = model.encode(m_ans, convert_to_tensor=True)
        emb2 = model.encode(s_ans, convert_to_tensor=True)
        cosine_scores = util.cos_sim(emb1, emb2)
        
        # Scale 0.0-1.0 to 0-10
        score = round(float(cosine_scores[0][0]) * 10, 1)
        
        return jsonify({"score": max(0, score)}) # Ensure no negative scores
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # This line tells the app to use the Port Railway provides
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port)