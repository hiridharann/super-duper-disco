import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from supabase import create_client, Client

# --- Page Config ---
st.set_page_config(page_title="Qwen Chat", page_icon="🚄")

# --- 1. Database & Auth Setup ---
# Load secrets from Streamlit Dashboard
try:
    SUPABASE_URL = st.secrets["SUPABASE_URL"]
    SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
    # If testing locally, use http://localhost:8501
    # On Streamlit Cloud, use your actual app URL (e.g. https://my-app.streamlit.app)
    REDIRECT_URL = st.secrets["REDIRECT_URL"] 
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
except Exception as e:
    st.error("Secrets not found! Please add SUPABASE_URL, SUPABASE_KEY, and REDIRECT_URL to Streamlit secrets.")
    st.stop()

# --- 2. Auth Logic ---
if "session" not in st.session_state:
    st.session_state.session = None

# Handle the return from Google (it adds ?code=... to the URL)
if "code" in st.query_params:
    try:
        code = st.query_params["code"]
        # Exchange code for a secure session
        session = supabase.auth.exchange_code_for_session({"auth_code": code})
        st.session_state.session = session
        st.query_params.clear() # Clean the URL
        st.rerun()
    except Exception as e:
        st.error(f"Login failed: {e}")

# Check if user is logged in
user = st.session_state.session.user if st.session_state.session else None

# --- 3. Login Screen (If not logged in) ---
if not user:
    st.title("🚄 Qwen 2.5 Login")
    st.write("Please sign in to access the AI.")
    
    try:
        # Get the Google Login URL
        res = supabase.auth.get_sign_in_url({
            "provider": "google",
            "redirect_to": REDIRECT_URL
        })
        st.link_button("Sign in with Google", res['url'], type="primary")
    except Exception as e:
        st.error(f"Auth Config Error: {e}")
    
    st.stop() # Stop here! Don't load the model yet.

# --- 4. Main App (Only runs if Logged In) ---
st.sidebar.success(f"Logged in as: {user.email}")
if st.sidebar.button("Logout"):
    supabase.auth.sign_out()
    st.session_state.session = None
    st.rerun()

st.title("🚄 Qwen 2.5 0.5B-Instruct")
st.caption("Running on CPU via Hugging Face Transformers")

# --- Load Model (Cached) ---
@st.cache_resource
def load_model():
    model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    with st.spinner("Downloading Qwen Model..."):
        tokenizer = AutoTokenizer.from_pretrained(model_id)
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

# --- Chat Interface ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask Qwen something..."):
    # 1. Show User Message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Generate Response
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        
        # Prepare Qwen Template
        chat_history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
        text = tokenizer.apply_chat_template(chat_history, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer([text], return_tensors="pt").to(model.device)

        with st.spinner("Thinking..."):
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response_tokens = outputs[0][len(inputs.input_ids[0]):]
        full_response = tokenizer.decode(response_tokens, skip_special_tokens=True)
        response_placeholder.markdown(full_response)
    
    st.session_state.messages.append({"role": "assistant", "content": full_response})

    # 3. Save to Supabase
    try:
        supabase.table("chat_logs").insert({
            "user_id": user.id,
            "email": user.email,
            "user_msg": prompt,
            "ai_msg": full_response,
            "model": "Qwen-0.5B"
        }).execute()
    except Exception as e:
        st.error(f"Failed to save to DB: {e}")
