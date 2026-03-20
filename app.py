from flask import Flask, request, jsonify
from flask_cors import CORS
from fastembed import TextEmbedding
import numpy as np
import os

app = Flask(__name__)
CORS(app)

# Lightweight model (33MB) suitable for Railway's resource constraints
model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")

@app.route('/get-score', methods=['POST'])
def get_score():
    try:
        data = request.json
        m_ans = data.get('model_answer', '')
        s_ans = data.get('student_answer', '')

        # Generate embeddings using fastembed (returns a generator, take first result)
        emb1 = np.array(list(model.query(m_ans))[0])
        emb2 = np.array(list(model.query(s_ans))[0])

        # Cosine similarity via numpy
        cosine_score = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

        # Scale 0.0-1.0 to 0-10
        score = round(float(cosine_score) * 10, 1)

        return jsonify({"score": max(0, score)})  # Ensure no negative scores
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # Use the PORT Railway provides
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port)