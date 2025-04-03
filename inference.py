from huggingface_hub import login
from google.colab import userdata

huggingface_token = userdata.get('huggingface_token')
login(token=huggingface_token)

import torch
import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

base_model = "meta-llama/Llama-2-7b-hf"
lora_path = "./lora_output"

# 4-bit quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

# load tokenizer
tokenizer = AutoTokenizer.from_pretrained(lora_path)
tokenizer.pad_token = tokenizer.eos_token  # required for decoder-only models

# load base model with quantization
base = AutoModelForCausalLM.from_pretrained(
    base_model,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float16
)

# load LoRA adapter on top
model = PeftModel.from_pretrained(base, lora_path)
model.eval()

# natural language to SQL inference function
def nl2sql(question):
    prompt = f"### NLQ:\n{question}\n\n### SQL:\n"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=30,
            temperature=0.7,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    sql = decoded.split("### SQL:")[-1].strip()
    return sql

# launch Gradio app
gr.Interface(
    fn=nl2sql,
    inputs=gr.Textbox(label="Enter a natural language query", placeholder="e.g. List all students enrolled after 2020"),
    outputs=gr.Textbox(label="Generated SQL query"),
    title="Text-to-SQL",
    description="Demo of a fine-tuned LLaMA-2-7b-hf model using LoRA on the Spider dataset."
).launch()