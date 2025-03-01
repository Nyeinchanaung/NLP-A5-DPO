# import streamlit as st
# from transformers import AutoModelForCausalLM, AutoTokenizer
# import torch

# # Streamlit app title and description
# st.title("A5: Optimization Human Preference")
# st.write("Enter a prompt below to generate a response using my DPO-trained model hosted on Hugging Face.")

# # Cache the model and tokenizer to avoid reloading on every interaction
# @st.cache_resource
# def load_model_and_tokenizer(repo_id):
#     try:
#         # Load model and tokenizer from Hugging Face Hub
#         model = AutoModelForCausalLM.from_pretrained(repo_id)
#         tokenizer = AutoTokenizer.from_pretrained(repo_id)
        
#         # Move model to GPU if available
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         model.to(device)
#         return model, tokenizer, device
#     except Exception as e:
#         st.error(f"Error loading model: {str(e)}")
#         return None, None, None

# # Replace with your actual Hugging Face repository ID
# repo_id = "nyeinchanaung/a5_dpo_model"  # Update this!
# model, tokenizer, device = load_model_and_tokenizer(repo_id)

# # Function to generate a response
# def generate_response(prompt, max_length=50):
#     if model is None or tokenizer is None:
#         return "Model failed to load. Check the repository or try again later."
    
#     try:
#         # Tokenize input
#         inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
#         # Generate response
#         outputs = model.generate(
#             **inputs,
#             max_length=max_length,
#             do_sample=True,    # Enable sampling for varied outputs
#             top_k=50,          # Top-k sampling for quality
#             top_p=0.95,        # Nucleus sampling
#             temperature=0.7,   # Control randomness
#             pad_token_id=tokenizer.eos_token_id  # Avoid padding warnings
#         )
        
#         # Decode and return response
#         response = tokenizer.decode(outputs[0], skip_special_tokens=True)
#         return response
#     except Exception as e:
#         return f"Error generating response: {str(e)}"

# # User input
# prompt = st.text_input("Enter your prompt:", "Hello, how are you?")

# # Generate button and response display
# if st.button("Generate Response"):
#     if not prompt.strip():
#         st.warning("Please enter a prompt!")
#     else:
#         with st.spinner("Generating response..."):
#             response = generate_response(prompt)
#             st.subheader("Response:")
#             st.write(response)

# # Footer with model info
# st.markdown("---")
# st.write(f"Model loaded from: [https://huggingface.co/{repo_id}](https://huggingface.co/{repo_id})")
# st.write("Built with ❤️ using Streamlit and Hugging Face Transformers")

import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Streamlit app setup
st.title("DPO Model Web App")
st.write("Enter a prompt to get a response from my DPO-trained model.")

# Cache model and tokenizer
@st.cache_resource
def load_model_and_tokenizer(repo_id):
    try:
        model = AutoModelForCausalLM.from_pretrained(repo_id)
        tokenizer = AutoTokenizer.from_pretrained(repo_id)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        return model, tokenizer, device
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None, None

# Replace with your Hugging Face repo ID
repo_id = "nyeinchanaung/a5_dpo_model"  # Update this!
model, tokenizer, device = load_model_and_tokenizer(repo_id)

# Generation function
def generate_response(prompt, max_new_tokens=32):  # Use max_new_tokens instead
    if model is None or tokenizer is None:
        return "Model failed to load."
    
    try:
        # Tokenize input
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        prompt_length = inputs["input_ids"].shape[-1]  # Length of prompt in tokens
        
        # Generate response
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,  # Only generate new tokens
            do_sample=True,
            top_k=50,
            top_p=0.9,
            temperature=0.7,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id
        )
        
        # Decode full output
        full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Remove prompt from the response
        if full_response.startswith(prompt):
            response = full_response[len(prompt):].strip()
        else:
            response = full_response  # Fallback if prompt isn’t at start
        
        # Trim to first sentence (optional)
        response = response.split(".")[0] + "." if "." in response else response
        return response
    except Exception as e:
        return f"Error: {str(e)}"

# User input
prompt = st.text_input("Enter your prompt:", "Hello, how are you?")
if st.button("Generate Response"):
    if not prompt.strip():
        st.warning("Please enter a prompt!")
    else:
        with st.spinner("Generating..."):
            response = generate_response(prompt)
            st.subheader("Response:")
            st.write(response)

# Footer
st.markdown("---")
st.write(f"Model: [https://huggingface.co/{repo_id}](https://huggingface.co/{repo_id})")