from transformers import pipeline
import torch

def save_model(model_path, model_name="McGill-NLP/Sheared-LLaMA-2.7B-weblinx"):
    # Determine the right device
    device = 0 if torch.cuda.is_available() else -1
    print(f"Using device: {'CUDA' if device == 0 else 'CPU'}")

    # Initialize the pipeline
    model = pipeline(model=model_name, device=device, torch_dtype=torch.float32)

    # Save the model and tokenizer
    model.model.save_pretrained(model_path)
    model.tokenizer.save_pretrained(model_path)

if __name__ == "__main__":
    save_model('./llam27_model')

