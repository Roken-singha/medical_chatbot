import os
import torch
from transformers import BertTokenizerFast, BertForQuestionAnswering  # Updated to Fast
from openvino.runtime import Core, Tensor
import numpy as np

# Paths
FINE_TUNED_MODEL_PATH = "F:/medical_chatbot/models/fine_tuned_biobert"
ONNX_MODEL_PATH = "F:/medical_chatbot/models/biobert_finetuned.onnx"
IR_MODEL_DIR = "F:/medical_chatbot/models"
IR_MODEL_XML = os.path.join(IR_MODEL_DIR, "biobert_finetuned.xml")
IR_MODEL_BIN = os.path.join(IR_MODEL_DIR, "biobert_finetuned.bin")

# Ensure output directory exists
os.makedirs(IR_MODEL_DIR, exist_ok=True)

# Load the fine-tuned BioBERT model and tokenizer
print("Loading fine-tuned BioBERT model...")
model = BertForQuestionAnswering.from_pretrained(FINE_TUNED_MODEL_PATH)
tokenizer = BertTokenizerFast.from_pretrained(FINE_TUNED_MODEL_PATH)  # Updated
model.eval()

# Move model to CPU for export
device = torch.device("cpu")
model.to(device)

# Export the BioBERT model to ONNX format
def export_to_onnx(model, onnx_path, tokenizer):
    print("Exporting model to ONNX format...")
    batch_size = 1
    seq_length = 512  # Updated to match training
    sample_question = "What are the treatments for prescription and illicit drug abuse?"
    sample_context = "Treatments include therapy, medication, and support groups."
    
    inputs = tokenizer(sample_question, sample_context, return_tensors="pt", padding="max_length", truncation="only_second", max_length=seq_length)
    dummy_input_ids = inputs["input_ids"]
    dummy_attention_mask = inputs["attention_mask"]
    dummy_token_type_ids = inputs["token_type_ids"]  # Added for consistency

    class BertWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
        
        def forward(self, input_ids, attention_mask, token_type_ids):  # Added token_type_ids
            outputs = self.model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            return outputs.start_logits, outputs.end_logits

    wrapped_model = BertWrapper(model)

    torch.onnx.export(
        wrapped_model,
        (dummy_input_ids, dummy_attention_mask, dummy_token_type_ids),  # Updated
        onnx_path,
        input_names=["input_ids", "attention_mask", "token_type_ids"],  # Updated
        output_names=["start_logits", "end_logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "token_type_ids": {0: "batch_size", 1: "sequence_length"},  # Added
            "start_logits": {0: "batch_size", 1: "sequence_length"},
            "end_logits": {0: "batch_size", 1: "sequence_length"},
        },
        opset_version=14,
        do_constant_folding=True,
    )
    print(f"ONNX model saved to {onnx_path}")

# Convert ONNX to OpenVINO IR
def convert_to_openvino_ir(onnx_path, ir_xml_path, ir_bin_path):
    print("Converting ONNX model to OpenVINO IR format...")
    from openvino import convert_model
    from openvino.runtime import serialize
    
    ir_model = convert_model(onnx_path)
    serialize(ir_model, ir_xml_path, ir_bin_path)
    print(f"OpenVINO IR model saved to {ir_xml_path} and {ir_bin_path}")

# Inference with OpenVINO
def inference_with_openvino(ir_xml_path, question, context, tokenizer, max_length=512):  # Updated max_length
    # Initialize OpenVINO Runtime
    core = Core()
    try:
        model = core.read_model(model=ir_xml_path)
        compiled_model = core.compile_model(model=model, device_name="CPU")
    except Exception as e:
        print(f"Error loading OpenVINO model: {e}")
        return ""

    # Tokenize question and context
    inputs = tokenizer(question, context, return_tensors="pt", padding="max_length", truncation="only_second", max_length=max_length)
    input_ids = inputs["input_ids"].cpu().numpy()
    attention_mask = inputs["attention_mask"].cpu().numpy()
    token_type_ids = inputs["token_type_ids"].cpu().numpy()  # Added
    
    if input_ids.size == 0:
        print("Error: Empty input sequence!")
        return ""

    # Create inference request
    infer_request = compiled_model.create_infer_request()
    infer_request.set_tensor("input_ids", Tensor(input_ids))
    infer_request.set_tensor("attention_mask", Tensor(attention_mask))
    infer_request.set_tensor("token_type_ids", Tensor(token_type_ids))  # Added
    
    # Perform inference
    infer_request.infer()
    
    # Get the output
    start_logits = infer_request.get_output_tensor(0).data
    end_logits = infer_request.get_output_tensor(1).data
    
    # Decode the result
    start_token = np.argmax(start_logits)
    end_token = np.argmax(end_logits)
    
    # Ensure valid token span
    if start_token > end_token or start_token < 0 or end_token >= max_length:
        print(f"Invalid token span: start={start_token}, end={end_token}. Returning empty.")
        return ""
    
    print(f"Start token: {start_token}, End token: {end_token}")
    
    # Extract and decode answer
    answer_tokens = input_ids[0, start_token:end_token + 1]
    answer = tokenizer.decode(answer_tokens, skip_special_tokens=True)
    
    # Avoid returning empty or invalid answers
    if not answer.strip():
        print("Answer is empty after decoding!")
        return "No valid answer found."
    
    print(f"Answer: {answer}")
    return answer

# Main execution
if __name__ == "__main__":
    # Export and convert if not already done
    if not os.path.exists(IR_MODEL_XML):
        export_to_onnx(model, ONNX_MODEL_PATH, tokenizer)
        convert_to_openvino_ir(ONNX_MODEL_PATH, IR_MODEL_XML, IR_MODEL_BIN)
    
    # Test inference
    question = "What are the symptoms of a cold?"
    context = "Common cold symptoms include a runny nose, sneezing, sore throat, and mild fever."
    answer = inference_with_openvino(IR_MODEL_XML, question, context, tokenizer, max_length=512)
    print(f"✅ Test inference complete! Answer: {answer}")