from flask import Flask, request, jsonify, render_template
from optimize import inference_with_openvino
from pathlib import Path
from transformers import BertTokenizerFast  # Updated
import pandas as pd
from difflib import SequenceMatcher

app = Flask(__name__, template_folder="../templates")

# Configuration
MODEL_PATH = Path("F:/medical_chatbot/models/biobert_finetuned.xml")  # Or update to pytorch_model.xml
TOKENIZER_PATH = "F:/medical_chatbot/models/fine_tuned_biobert"
DATASET_PATH = "F:/medical_chatbot/data/processed/processed_data.csv"  

# Verify model exists
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"❌ OpenVINO model not found at {MODEL_PATH}")

# Load tokenizer
tokenizer = BertTokenizerFast.from_pretrained(TOKENIZER_PATH)  # Updated

# Load dataset
if not Path(DATASET_PATH).exists():
    raise FileNotFoundError(f"❌ Dataset not found at {DATASET_PATH}")
data = pd.read_csv(DATASET_PATH)
print(f"✅ Loaded dataset with {len(data)} questions.")

def find_context(user_question, dataset, threshold=0.8):
    best_match = None
    highest_similarity = 0.0
    for index, row in dataset.iterrows():
        similarity = SequenceMatcher(None, user_question.lower(), row["questions"].lower()).ratio()  # Updated to "questions"
        if similarity > highest_similarity:
            highest_similarity = similarity
            best_match = row["context"]
    if highest_similarity >= threshold:
        print(f"📏 Matched question with similarity {highest_similarity:.2f}")
        return best_match
    else:
        print(f"⚠️ No good match found (best similarity: {highest_similarity:.2f})")
        return "I don’t have enough information to answer this question."

def get_response(question):
    try:
        if not question or not isinstance(question, str):
            return "⚠️ Invalid input."
        print(f"📥 Received question: {question}")
        context = find_context(question, data)
        response = inference_with_openvino(MODEL_PATH, question, context, tokenizer, max_length=512)  # Updated max_length
        if not response:
            print("⚠️ OpenVINO returned an empty response!")
            return "I don’t have a response for that."
        print(f"🤖 Chatbot response: {response}")
        return response
    except Exception as e:
        print(f"❌ Error in get_response: {str(e)}")
        return "Error processing your question."

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/generate", methods=["POST"])
def chat():
    try:
        data = request.get_json()
        print(f"📥 Received request data: {data}")
        if not data or "question" not in data:
            return jsonify({"error": "No question provided"}), 400
        question = data["question"].strip()
        if not question:
            return jsonify({"error": "Empty question"}), 400
        response = get_response(question)
        return jsonify({"response": response})
    except Exception as e:
        print(f"❌ Server error: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)