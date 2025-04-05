import os
import json
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize

# Download required NLTK data
nltk.download('punkt')

# Define file paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JSON_PATH = os.path.join(BASE_DIR, "data", "raw", "medical_intents.json")
CSV_PATH = os.path.join(BASE_DIR, "data", "raw", "MedQuAD.csv")
OUTPUT_PATH = os.path.join(BASE_DIR, "data", "processed", "processed_data.csv")

def preprocess_text(text):
    """Tokenize and clean text, handling None or non-string inputs."""
    if text is None or pd.isna(text):
        return ""
    return " ".join(word_tokenize(str(text).lower()))

def load_json_data(file_path):
    """Load and process data from medical_intents.json"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"JSON file not found: {file_path}")
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)

        qa_pairs = []
        for intent in data.get("intents", []):
           
            
            # Extracting the Question and Answer directly from the JSON
            question = intent.get("Question", "no_question")  # Default if 'Question' key is missing
            answer = intent.get("Answer", "no_answer")  # Default if 'Answer' key is missing
            
            # 'Context' field contains both the context and the answer, but we are treating it as context
            context = intent.get("Context", "no_context")  # Default if 'Context' key is missing
            
            # Add preprocessed (question, response, context) tuples
            qa_pairs.append((preprocess_text(question), preprocess_text(answer), preprocess_text(context)))

        return qa_pairs

    except (json.JSONDecodeError, KeyError) as e:
        print("Error loading JSON:", e)
        return []


 
def load_csv_data(file_path):
    """Load and process data from medquad.csv"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"CSV file not found: {file_path}")
    try:
        df = pd.read_csv(file_path)
        print("CSV Loaded Successfully. Columns:", df.columns)  # Debugging print
        
        # Normalize column names (remove spaces, lowercase)
        df.columns = df.columns.str.strip().str.lower()

        # Check if required columns are present ('question', 'answer', 'context')
        if "question" in df.columns and "answer" in df.columns:
            # Handle missing 'context' column by defaulting to 'no_context'
            if "context" not in df.columns:
                df["context"] = "no_context"
            
            # Handle missing context values
            df["context"] = df["context"].fillna("no_context")  # Ensure no NaN in the context column
            
            return [(preprocess_text(q), preprocess_text(a), preprocess_text(c)) 
                    for q, a, c in zip(df["question"], df["answer"], df["context"])]
        else:
            print("CSV file is missing required columns: 'question', 'answer', and 'context'")
            return []
    except pd.errors.ParserError as e:
        print("Error loading CSV:", e)
        return []

def save_preprocessed_data(qa_pairs, output_path):
    """Save processed data to a CSV file"""
    if qa_pairs:
        df = pd.DataFrame(qa_pairs, columns=["questions", "answers", "context"])
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"✅ Preprocessed data saved to: {output_path}")
    else:
        print("No data to save.")

if __name__ == "__main__":
    try:
        # Load data from JSON and CSV files
        json_data = load_json_data(JSON_PATH)
        csv_data = load_csv_data(CSV_PATH)
        
        # Combine the data from both sources
        all_data = json_data + csv_data
        
        # Save the combined data to CSV
        save_preprocessed_data(all_data, OUTPUT_PATH)
        print("Data preprocessing completed successfully.")
    except Exception as e:
        print(f"Error: {e}")
