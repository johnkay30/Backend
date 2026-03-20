from flask import Flask, request, jsonify
from flask_cors import CORS
from fastembed import TextEmbedding
import numpy as np
import os

app = Flask(__name__)
CORS(app)

# 1. Load the lightweight model (approx 33MB)
# This replaces SentenceTransformer
model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")

def calculate_similarity(vec1, vec2):
    # 2. Manual Cosine Similarity (Replaces util.cos_sim)
    # This avoids importing the heavy 'sentence_transformers.util'
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    return dot_product / (norm1 * norm2)

@app.route('/get-score', methods=['POST'])
def get_score():
    try:
        data = request.json
        m_ans = data.get('model_answer', '')
        s_ans = data.get('student_answer', '')

        if not m_ans or not s_ans:
            return jsonify({"score": 0.0}), 400

        # 3. Generate embeddings (returns a generator, so we convert to list)
        embeddings = list(model.embed([m_ans, s_ans]))
        
        # 4. Calculate similarity and scale to 0-10
        raw_sim = calculate_similarity(embeddings[0], embeddings[1])
        final_score = round(float(raw_sim) * 10, 1)

        # Ensure score stays within 0-10 range
        return jsonify({"score": max(0, min(10, final_score))})

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # 5. Dynamic Port for Railway/Render
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port)