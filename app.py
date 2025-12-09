import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- Page Config ---
st.set_page_config(page_title="Qwen 0.5B Chat", page_icon="🚄")
st.title("🚄 Qwen 2.5 0.5B-Instruct")
st.caption("Running on CPU via Hugging Face Transformers")

# --- Load Model (Cached) ---
# We use @st.cache_resource so it only loads once!
@st.cache_resource
def load_model():
    model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    
    with st.spinner("Downloading Model... (approx 1GB)"):
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        
        # We load in float32 because float16 is often unstable on pure CPUs
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
            device_map="cpu",
            low_cpu_mem_usage=True
        )
    return tokenizer, model

try:
    tokenizer, model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# --- Chat History ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous chats
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Chat Input ---
if prompt := st.chat_input("Ask Qwen something..."):
    # 1. Display User Message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Generate Response
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        
        # Prepare Inputs
        # We re-format the history into Qwen's chat template
        chat_history = [
            {"role": m["role"], "content": m["content"]} 
            for m in st.session_state.messages
        ]
        
        text = tokenizer.apply_chat_template(
            chat_history,
            tokenize=False,
            add_generation_prompt=True
        )
        
        inputs = tokenizer([text], return_tensors="pt").to(model.device)

        # Generate (No streaming in basic Transformers on CPU to keep it simple)
        with st.spinner("Thinking..."):
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode
        # We slice the output to remove the input prompt
        response_tokens = outputs[0][len(inputs.input_ids[0]):]
        full_response = tokenizer.decode(response_tokens, skip_special_tokens=True)
        
        response_placeholder.markdown(full_response)
    
    # 3. Save Assistant Message
    st.session_state.messages.append({"role": "assistant", "content": full_response})
