import os
import json
from flask import Flask, request, jsonify
from transformers import TFGPT2LMHeadModel, GPT2Tokenizer
import gradio as gr

app = Flask(__name__)

# Local cache directory
LOCAL_MODEL_DIR = "./models/gpt2"

# Load or download model
if os.path.exists(LOCAL_MODEL_DIR):
    print(f"🔍 Loading GPT-2 model from local directory: {LOCAL_MODEL_DIR}")
    tokenizer = GPT2Tokenizer.from_pretrained(LOCAL_MODEL_DIR)
    model = TFGPT2LMHeadModel.from_pretrained(LOCAL_MODEL_DIR)
else:
    print("⬇️ Downloading GPT-2 model from Hugging Face...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = TFGPT2LMHeadModel.from_pretrained("gpt2", from_pt=True)
    os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)
    tokenizer.save_pretrained(LOCAL_MODEL_DIR)
    model.save_pretrained(LOCAL_MODEL_DIR)


def generate_text(prompt, max_new_tokens=1000, temperature=0.7):
    inputs = tokenizer(prompt, return_tensors="tf")
    outputs = model.generate(
        inputs["input_ids"],
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=True,
    )
    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Remove the original prompt from the beginning
    generated_only = full_text[len(prompt):].strip()

    # return {
    #     "prompt": prompt,
    #     "generated_text": generated_only,
    #     "full_output": full_text,  # if you still want the raw combined version
    #     "max_new_tokens": max_new_tokens,
    #     "temperature": temperature,
    # }
    return full_text
    

@app.route("/generate", methods=["POST"])
def generate_post():
    data = request.json
    business_type = data.get("business_type", "Business")
    platform = data.get("platform", "Facebook")

    prompt = data.get("prompt", f"""
    You are an expert Social Media Marketer and professional content writer...
    For a {business_type}, write an engaging post optimized for {platform}.
    """)

    result = generate_text(prompt, max_new_tokens=600)  # bump tokens for ~500 words
    return jsonify(result)



# --- Gradio UI ---
def gradio_generate(business_type, platform, prompt):
    with app.test_client() as client:
        response = client.post(
            "/generate",
            json={"business_type": business_type, "platform": platform, "prompt": prompt},
        )
        return json.dumps(response.get_json(), indent=2)


demo = gr.Interface(
    fn=gradio_generate,
    inputs=[
        gr.Textbox(label="Business Type", value="Coffee Shop"),
        gr.Dropdown(["Facebook", "Instagram", "LinkedIn"], value="Facebook", label="Platform"),
        gr.Textbox(
            label="Custom Prompt (optional)", 
            placeholder="Leave empty to auto-generate based on Business + Platform",
            lines=6,
            value="You are a Social Media Marketer and expert content writer. For a Coffee Shop, write an engaging post optimized for Facebook."
        ),
    ],
    outputs=gr.Code(label="Generated JSON Output", language="json"),
    title="🚀 Social Media Post Generator"
)


@app.route("/ui")
def ui():
    return demo.launch(
        inline=True,
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        prevent_thread_lock=True,
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
