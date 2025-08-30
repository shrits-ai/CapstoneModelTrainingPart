import torch
import gradio as gr
from model import CustomLLM, CustomConfig
from transformers import AutoTokenizer

class ModelLoader:
    def __init__(self):
        # Load config
        self.config = CustomConfig()
        # Instantiate model
        self.model = CustomLLM(self.config)
        
        # Load trained weights
        state_dict = torch.load('pytorch_model.bin', map_location='cpu')
        self.model.load_state_dict(state_dict)
        self.model.eval()
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/cosmo2-tokenizer")
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate(self, prompt, max_new_tokens=100, temperature=0.9, top_k=50, top_p=0.95):
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs.input_ids
        
        with torch.no_grad():
            generated = self.model.generate(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                eos_token_id=None,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        return self.tokenizer.decode(generated[0], skip_special_tokens=True)

# Initialize model
loader = ModelLoader()

# Create Gradio interface
interface = gr.Interface(
    fn=loader.generate,
    inputs=[
        gr.Textbox(lines=4, label="Input Prompt"),
        gr.Slider(1, 500, value=100, label="Max New Tokens"),
        gr.Slider(0.1, 2.0, value=0.9, label="Temperature"),
        gr.Slider(1, 100, value=50, label="Top K"),
        gr.Slider(0.1, 1.0, value=0.95, label="Top P")
    ],
    outputs=gr.Textbox(label="Generated Output"),
    title="Custom LLM Demo",
    description="Generate text using your custom-trained LLM"
)

interface.launch(share=True)
