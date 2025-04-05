import pandas as pd
from datasets import Dataset
from transformers import BertTokenizerFast, BertForQuestionAnswering, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
import torch

# Load dataset
dataset_path = r'F:\medical_chatbot\data\processed\processed_data.csv'
df = pd.read_csv(dataset_path)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Training on {device}.")

# Initialize tokenizer (use BertTokenizerFast for offset mapping)
tokenizer = BertTokenizerFast.from_pretrained("dmis-lab/biobert-base-cased-v1.1")

# Function to find answer positions
def find_answer_positions(example):
    """Find start and end token positions of the answer in the context."""
    context = str(example["context"])
    answer = str(example["answers"]) if "answers" in example else ""
    
    # Tokenize questions and context together
    encoding = tokenizer(example["questions"], context, max_length=400, truncation="only_second", padding="max_length", return_offsets_mapping=True)
    input_ids = encoding["input_ids"]
    offsets = encoding["offset_mapping"]
    
    # Find answer span in context
    start_char = context.find(answer)
    if start_char == -1 or not answer:
        return 0, 0
    
    end_char = start_char + len(answer)
    start_token, end_token = 0, 0
    
    for idx, (start, end) in enumerate(offsets):
        if start <= start_char < end:
            start_token = idx
        if start < end_char <= end:
            end_token = idx
            break
    
    if start_token == 0 or end_token == 0:
        return 0, 0
    
    return start_token, end_token

# Preprocess dataset
def preprocess(example):
    encoding = tokenizer(example["questions"], example["context"], max_length=400, truncation="only_second", padding="max_length", return_tensors="pt")
    start_pos, end_pos = find_answer_positions(example)
    return {
        "input_ids": encoding["input_ids"][0],
        "attention_mask": encoding["attention_mask"][0],
        "token_type_ids": encoding["token_type_ids"][0],
        "start_positions": start_pos,
        "end_positions": end_pos
    }

# Split and convert to Dataset
train_df, eval_df = train_test_split(df, test_size=0.2, random_state=42)
train_dataset = Dataset.from_pandas(train_df).map(preprocess)
eval_dataset = Dataset.from_pandas(eval_df).map(preprocess)

# Format for PyTorch
train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "token_type_ids", "start_positions", "end_positions"])
eval_dataset.set_format("torch", columns=["input_ids", "attention_mask", "token_type_ids", "start_positions", "end_positions"])

# Load model
model = BertForQuestionAnswering.from_pretrained("dmis-lab/biobert-base-cased-v1.1")
model.to(device)

# Training arguments
training_args = TrainingArguments(
    output_dir=r"F:\medical_chatbot\models\fine_tuned_biobert",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_steps=10,
    save_steps=500,
    save_total_limit=2,
    evaluation_strategy="steps",
    eval_steps=500,
    save_strategy="steps",
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

# Train
trainer.train()

# Save model and tokenizer
model.save_pretrained(r"F:\medical_chatbot\models\fine_tuned_biobert")
tokenizer.save_pretrained(r"F:\medical_chatbot\models\fine_tuned_biobert")