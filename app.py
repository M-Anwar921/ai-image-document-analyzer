import streamlit as st
from dotenv import load_dotenv
from groq import Groq
import base64
import os

load_dotenv()

# ---------------------------------------------------------------
# GROQ CLIENT
# ---------------------------------------------------------------
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))


# ---------------------------------------------------------------
# IMAGE ANALYSIS FUNCTION
# Converts image to base64 and sends to Groq vision model
# base64 = a way to convert image bytes into text so API can read it
# ---------------------------------------------------------------
def analyze_image(image_file):
    # Read image bytes
    image_bytes = image_file.read()
    
    # Convert to base64 string
    # This is how we send images through a text-based API
    base64_image = base64.b64encode(image_bytes).decode("utf-8")
    
    # Detect image type from filename
    filename = image_file.name.lower()
    if filename.endswith(".png"):
        media_type = "image/png"
    elif filename.endswith(".webp"):
        media_type = "image/webp"
    elif filename.endswith(".gif"):
        media_type = "image/gif"
    else:
        media_type = "image/jpeg"
    
    try:
        response = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            # Send the actual image as base64
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{media_type};base64,{base64_image}"
                            }
                        },
                        {
                            # Instructions for AI on what to analyze
                            "type": "text",
                            "text": """Analyze this image thoroughly and provide:

1. **Overview** — What is this image about in 2-3 sentences?
2. **Key Elements** — List the main objects, people, or subjects visible
3. **Colors & Style** — Describe the dominant colors and visual style
4. **Context & Setting** — Where was this taken or what is the context?
5. **Interesting Details** — Any notable or unusual details worth mentioning
6. **Possible Use Case** — What might this image be used for?

Be specific, insightful and helpful."""
                        }
                    ]
                }
            ],
            max_tokens=1024
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"❌ Analysis failed: {str(e)}"


# ---------------------------------------------------------------
# PDF TEXT EXTRACTION FUNCTION
# pdfplumber opens the PDF and extracts all text page by page
# ---------------------------------------------------------------
def extract_pdf_text(pdf_file):
    import pdfplumber
    
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for i, page in enumerate(pdf.pages):
            page_text = page.extract_text()
            if page_text:
                text += f"\n--- Page {i+1} ---\n{page_text}"
    
    return text.strip()

# ---------------------------------------------------------------
# DOCUMENT ANALYSIS FUNCTION
# Sends extracted PDF text to Groq for intelligent analysis
# ---------------------------------------------------------------
def analyze_document(pdf_file):
    # Step 1 — Extract text from PDF
    extracted_text = extract_pdf_text(pdf_file)
    
    # Check if we got any text
    if not extracted_text:
        return "❌ Could not extract text from this PDF. It may be scanned or image-based."
    
    # Step 2 — Limit text length to avoid token limits
    # Groq has a max input size so we trim very long documents
    MAX_CHARS = 8000
    if len(extracted_text) > MAX_CHARS:
        extracted_text = extracted_text[:MAX_CHARS] + "\n\n[Document trimmed due to length...]"
    
    # Step 3 — Send to Groq for analysis
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": """You are an expert document analyzer. 
                    When given document text, provide a thorough analysis 
                    in a clear, structured format."""
                },
                {
                    "role": "user",
                    "content": f"""Analyze this document and provide:

1. **Document Type** — What kind of document is this?
2. **Summary** — A clear 3-4 sentence summary of the main content
3. **Key Points** — The 5 most important points or findings
4. **Important Details** — Dates, names, numbers, or facts worth noting
5. **Tone & Purpose** — What is the purpose of this document?
6. **Action Items** — What should the reader do after reading this?

Document text:
{extracted_text}"""
                }
            ],
            max_tokens=1024,
            temperature=0.3
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"❌ Analysis failed: {str(e)}"
    

    # ---------------------------------------------------------------
# FOLLOW-UP CHAT FUNCTION
# Allows user to ask questions about the analyzed file
# context = the original analysis result (image description or doc text)
# chat_history = previous questions and answers in this session
# ---------------------------------------------------------------
def ask_followup(question, context, chat_history, file_type="document"):
    # Build conversation with full context
    messages = [
        {
            "role": "system",
            "content": f"""You are an intelligent assistant helping a user 
            understand a {file_type} they uploaded. 
            You have already analyzed it and provided an initial analysis.
            
            Here is the content/analysis of the {file_type}:
            {context}
            
            Now answer the user's follow-up questions about this {file_type}.
            Be specific, helpful and refer to the actual content when answering.
            Keep answers concise — 2-3 paragraphs maximum."""
        }
    ]
    
    # Add chat history so AI remembers previous questions
    for msg in chat_history[-6:]:  # Last 6 messages for memory
        messages.append({
            "role": msg["role"],
            "content": msg["content"]
        })
    
    # Add current question
    messages.append({
        "role": "user",
        "content": question
    })
    
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            max_tokens=512,
            temperature=0.5
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"❌ Error: {str(e)}"
    

# ---------------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------------
st.set_page_config(
    page_title="AI Analyzer",
    page_icon="🔍",
    layout="wide"
)

# ---------------------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------------------
st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Page background */
    .stApp { background: #0A0A0A; }

    /* Remove default padding */
    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 1rem !important;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        background: #111111 !important;
        border-radius: 12px !important;
        padding: 4px !important;
        gap: 4px !important;
        border: 1px solid #1E1E1E !important;
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent !important;
        border-radius: 8px !important;
        color: #666666 !important;
        font-weight: 500 !important;
        padding: 8px 24px !important;
    }
    .stTabs [aria-selected="true"] {
        background: #1A1A1A !important;
        color: #00FF88 !important;
    }
    .stTabs [data-baseweb="tab-border"] {
        display: none !important;
    }

    /* Upload zone */
    .upload-zone {
        border: 1.5px dashed #2D2D2D;
        border-radius: 16px;
        padding: 60px 20px;
        text-align: center;
        background: #111111;
        transition: border-color 0.2s;
    }
    .upload-zone:hover { border-color: #00FF88; }

    /* Result card */
    .result-card {
        background: #111111;
        border: 1px solid #1E1E1E;
        border-radius: 14px;
        padding: 20px 24px;
        margin-bottom: 12px;
    }
    .result-label {
        font-size: 11px;
        font-weight: 500;
        color: #00FF88;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-bottom: 8px;
    }
    .result-text {
        font-size: 14px;
        color: #CCCCCC;
        line-height: 1.7;
    }

    /* Analyze button */
    .stButton > button {
        background: #00FF88 !important;
        color: #000000 !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 12px 32px !important;
        font-weight: 700 !important;
        font-size: 15px !important;
        width: 100% !important;
        transition: opacity 0.2s !important;
    }
    .stButton > button:hover { opacity: 0.85 !important; }

    /* File uploader */
    [data-testid="stFileUploaderDropzone"] {
        background: #111111 !important;
        border: 1.5px dashed #2D2D2D !important;
        border-radius: 14px !important;
        padding: 30px !important;
    }

    /* Status indicator */
    .status-online {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: #0D1F0D;
        border: 1px solid #1A3A1A;
        border-radius: 20px;
        padding: 5px 14px;
        font-size: 12px;
        font-weight: 600;
        color: #00FF88;
        letter-spacing: 0.5px;
    }
    .dot {
        width: 7px;
        height: 7px;
        background: #00FF88;
        border-radius: 50%;
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.3; }
    }

    hr { border-color: #1E1E1E !important; }

    /* Scrollbar */
    ::-webkit-scrollbar { width: 4px; }
    ::-webkit-scrollbar-track { background: #0A0A0A; }
    ::-webkit-scrollbar-thumb { background: #222222; border-radius: 4px; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------
# SESSION STATE
# ---------------------------------------------------------------
if "image_result" not in st.session_state:
    st.session_state.image_result = None
if "doc_result" not in st.session_state:
    st.session_state.doc_result = None

# Chat histories for follow-up questions
if "image_chat" not in st.session_state:
    st.session_state.image_chat = []
if "doc_chat" not in st.session_state:
    st.session_state.doc_chat = []

# Store extracted content for chat context
if "image_context" not in st.session_state:
    st.session_state.image_context = None
if "doc_context" not in st.session_state:
    st.session_state.doc_context = None

# Input counters for clearing input fields
if "img_input_counter" not in st.session_state:
    st.session_state.img_input_counter = 0
if "doc_input_counter" not in st.session_state:
    st.session_state.doc_input_counter = 0

# ---------------------------------------------------------------
# HEADER
# ---------------------------------------------------------------
col_left, col_center, col_right = st.columns([1, 2, 1])

with col_center:
    # Status bar
    st.markdown("""
    <div style="text-align:center; margin-bottom:16px;">
        <span class="status-online">
            <span class="dot"></span>
            AI ONLINE
        </span>
    </div>
    """, unsafe_allow_html=True)

    # Title
    st.markdown("""
    <div style="text-align:center; margin-bottom:24px;">
        <div style="font-size:32px; font-weight:800; color:#FFFFFF; 
                    letter-spacing:2px; margin-bottom:8px;">
            AI <span style="color:#00FF88;">ANALYZER</span>
        </div>
        <div style="font-size:14px; color:#555555; line-height:1.6;">
            Upload any image or document.<br>
            Get instant AI-powered insights.
        </div>
    </div>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------------
# MAIN TABS — Image vs Document
# ---------------------------------------------------------------
tab1, tab2 = st.tabs(["🖼️  Image Analysis", "📄  Document Analysis"])

# =====================
# TAB 1 — IMAGE
# =====================
with tab1:
    st.markdown("###")

    left, right = st.columns([1, 1], gap="large")

    with left:
        st.markdown("""
        <div style="font-size:13px; font-weight:500; 
                    color:#888888; margin-bottom:12px; 
                    text-transform:uppercase; letter-spacing:1px;">
            Upload Image
        </div>
        """, unsafe_allow_html=True)

        # Image uploader
        uploaded_image = st.file_uploader(
            label="Upload image",
            type=["jpg", "jpeg", "png", "webp", "gif"],
            label_visibility="collapsed",
            key="image_upload"
        )

        if uploaded_image:
            # Show preview
            st.image(uploaded_image, caption="Uploaded image", use_container_width=True)
            st.success(f"✅ {uploaded_image.name} ready for analysis")

            # Analyze button
            if st.button("🔍 Analyze Image", key="analyze_img"):
                uploaded_image.seek(0)
                with st.spinner("🤖 AI is analyzing your image..."):
                    result = analyze_image(uploaded_image)
                    st.session_state.image_result = result
                    # Save context and reset chat for new image
                    st.session_state.image_context = result
                    st.session_state.image_chat = []
                    st.rerun()
        else:
            st.markdown("""
            <div style="text-align:center; padding:40px 20px; 
                        color:#444444; font-size:13px;">
                <div style="font-size:40px; margin-bottom:12px;">🖼️</div>
                <div style="color:#666666;">Supported: JPG, PNG, WEBP, GIF</div>
                <div style="color:#444444; margin-top:4px; font-size:12px;">
                    Max size: 10MB
                </div>
            </div>
            """, unsafe_allow_html=True)

    with right:
        st.markdown("""
        <div style="font-size:13px; font-weight:500; 
                    color:#888888; margin-bottom:12px; 
                    text-transform:uppercase; letter-spacing:1px;">
            Analysis Results
        </div>
        """, unsafe_allow_html=True)
        if st.session_state.image_result:
            st.markdown("""
            <div class="result-label">AI ANALYSIS</div>
            """, unsafe_allow_html=True)
            st.markdown(st.session_state.image_result)

            # ---------------------------------------------------------------
            # FOLLOW-UP CHAT — Image
            # ---------------------------------------------------------------
            st.markdown("---")
            st.markdown("""
            <div style="font-size:13px; font-weight:500; color:#00FF88; 
                        text-transform:uppercase; letter-spacing:1px; 
                        margin-bottom:10px;">
                Ask about this image
            </div>
            """, unsafe_allow_html=True)

            # Show chat history
            for msg in st.session_state.image_chat:
                if msg["role"] == "user":
                    st.markdown(f"""
                    <div style="background:#1A1A1A; border-radius:10px; 
                                padding:10px 14px; margin-bottom:8px;
                                border-left:3px solid #00FF88; font-size:13px;
                                color:#CCCCCC;">
                        <span style="color:#00FF88; font-size:11px; 
                                     font-weight:600;">YOU</span><br>
                        {msg["content"]}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div style="background:#111111; border-radius:10px; 
                                padding:10px 14px; margin-bottom:8px;
                                border-left:3px solid #333333; font-size:13px;
                                color:#CCCCCC;">
                        <span style="color:#666666; font-size:11px; 
                                     font-weight:600;">AI</span><br>
                        {msg["content"]}
                    </div>
                    """, unsafe_allow_html=True)

            # Input for follow-up question
            with st.form(key=f"img_chat_{st.session_state.img_input_counter}",
                        clear_on_submit=True):
                img_q_col, img_btn_col = st.columns([5, 1])
                with img_q_col:
                    img_question = st.text_input(
                        label="Ask",
                        placeholder="Ask anything about this image...",
                        label_visibility="collapsed"
                    )
                with img_btn_col:
                    img_send = st.form_submit_button("➤")

            if img_send and img_question.strip():
                # Add user question to history
                st.session_state.image_chat.append({
                    "role": "user",
                    "content": img_question
                })
                # Get AI answer
                with st.spinner("🤔 Thinking..."):
                    answer = ask_followup(
                        img_question,
                        st.session_state.image_context,
                        st.session_state.image_chat,
                        file_type="image"
                    )
                # Add AI answer to history
                st.session_state.image_chat.append({
                    "role": "assistant",
                    "content": answer
                })
                st.session_state.img_input_counter += 1
                st.rerun()

        else:
            st.markdown("""
            <div style="text-align:center; padding:60px 20px;
                        border:1.5px dashed #1E1E1E; border-radius:14px;
                        color:#444444;">
                <div style="font-size:36px; margin-bottom:10px;">✨</div>
                <div style="color:#555555; font-size:14px;">
                    Results will appear here
                </div>
                <div style="color:#333333; font-size:12px; margin-top:4px;">
                    Upload an image and click Analyze
                </div>
            </div>
            """, unsafe_allow_html=True)

# =====================
# TAB 2 — DOCUMENT
# =====================
with tab2:
    st.markdown("###")

    left2, right2 = st.columns([1, 1], gap="large")

    with left2:
        st.markdown("""
        <div style="font-size:13px; font-weight:500; 
                    color:#888888; margin-bottom:12px; 
                    text-transform:uppercase; letter-spacing:1px;">
            Upload Document
        </div>
        """, unsafe_allow_html=True)

        # PDF uploader
        uploaded_doc = st.file_uploader(
            label="Upload document",
            type=["pdf"],
            label_visibility="collapsed",
            key="doc_upload"
        )

        if uploaded_doc:
            # Show file info
            file_size = uploaded_doc.size / 1024
            st.markdown(f"""
            <div style="background:#111111; border:1px solid #1E1E1E; 
                        border-radius:12px; padding:16px 20px; margin-bottom:16px;">
                <div style="font-size:32px; text-align:center; margin-bottom:10px;">📄</div>
                <div style="font-size:14px; font-weight:500; 
                            color:#FFFFFF; text-align:center;">
                    {uploaded_doc.name}
                </div>
                <div style="font-size:12px; color:#555555; text-align:center; margin-top:4px;">
                    {file_size:.1f} KB
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.success(f"✅ Document ready for analysis")

            # Analyze button
            if st.button("🔍 Analyze Document", key="analyze_doc"):
                with st.spinner("📄 Extracting text and analyzing..."):
                    uploaded_doc.seek(0)
                    # Extract text separately for chat context
                    doc_text = extract_pdf_text(uploaded_doc)
                    uploaded_doc.seek(0)
                    result = analyze_document(uploaded_doc)
                    st.session_state.doc_result = result
                    # Save full text as context for follow-up questions
                    st.session_state.doc_context = doc_text[:6000]
                    st.session_state.doc_chat = []
                    st.rerun()
        else:
            st.markdown("""
            <div style="text-align:center; padding:40px 20px; 
                        color:#444444; font-size:13px;">
                <div style="font-size:40px; margin-bottom:12px;">📄</div>
                <div style="color:#666666;">Supported: PDF files only</div>
                <div style="color:#444444; margin-top:4px; font-size:12px;">
                    Max size: 10MB
                </div>
            </div>
            """, unsafe_allow_html=True)

    with right2:
        st.markdown("""
        <div style="font-size:13px; font-weight:500; 
                    color:#888888; margin-bottom:12px; 
                    text-transform:uppercase; letter-spacing:1px;">
            Analysis Results
        </div>
        """, unsafe_allow_html=True)

        if st.session_state.doc_result:
            st.markdown("""
            <div class="result-label">DOCUMENT ANALYSIS</div>
            """, unsafe_allow_html=True)
            st.markdown(st.session_state.doc_result)

            # ---------------------------------------------------------------
            # FOLLOW-UP CHAT — Document
            # ---------------------------------------------------------------
            st.markdown("---")
            st.markdown("""
            <div style="font-size:13px; font-weight:500; color:#00FF88;
                        text-transform:uppercase; letter-spacing:1px;
                        margin-bottom:10px;">
                Ask about this document
            </div>
            """, unsafe_allow_html=True)

            # Show chat history
            for msg in st.session_state.doc_chat:
                if msg["role"] == "user":
                    st.markdown(f"""
                    <div style="background:#1A1A1A; border-radius:10px;
                                padding:10px 14px; margin-bottom:8px;
                                border-left:3px solid #00FF88; font-size:13px;
                                color:#CCCCCC;">
                        <span style="color:#00FF88; font-size:11px;
                                     font-weight:600;">YOU</span><br>
                        {msg["content"]}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div style="background:#111111; border-radius:10px;
                                padding:10px 14px; margin-bottom:8px;
                                border-left:3px solid #333333; font-size:13px;
                                color:#CCCCCC;">
                        <span style="color:#666666; font-size:11px;
                                     font-weight:600;">AI</span><br>
                        {msg["content"]}
                    </div>
                    """, unsafe_allow_html=True)

            # Input for follow-up question
            with st.form(key=f"doc_chat_{st.session_state.doc_input_counter}",
                        clear_on_submit=True):
                doc_q_col, doc_btn_col = st.columns([5, 1])
                with doc_q_col:
                    doc_question = st.text_input(
                        label="Ask",
                        placeholder="Ask anything about this document...",
                        label_visibility="collapsed"
                    )
                with doc_btn_col:
                    doc_send = st.form_submit_button("➤")

            if doc_send and doc_question.strip():
                # Add user question to history
                st.session_state.doc_chat.append({
                    "role": "user",
                    "content": doc_question
                })
                # Get AI answer with document context
                with st.spinner("🤔 Thinking..."):
                    answer = ask_followup(
                        doc_question,
                        st.session_state.doc_context,
                        st.session_state.doc_chat,
                        file_type="document"
                    )
                # Add AI answer to history
                st.session_state.doc_chat.append({
                    "role": "assistant",
                    "content": answer
                })
                st.session_state.doc_input_counter += 1
                st.rerun()
        else:
            st.markdown("""
            <div style="text-align:center; padding:60px 20px;
                        border:1.5px dashed #1E1E1E; border-radius:14px;
                        color:#444444;">
                <div style="font-size:36px; margin-bottom:10px;">✨</div>
                <div style="color:#555555; font-size:14px;">
                    Results will appear here
                </div>
                <div style="color:#333333; font-size:12px; margin-top:4px;">
                    Upload a PDF and click Analyze
                </div>
            </div>
            """, unsafe_allow_html=True)

# ---------------------------------------------------------------
# FOOTER
# ---------------------------------------------------------------
st.markdown("---")
st.markdown("""
<div style="text-align:center; font-size:11px; color:#333333; padding:4px 0;">
    AI Analyzer · Built by Anwar · Powered by Groq AI · 
    Supports Images & PDF Documents
</div>
""", unsafe_allow_html=True)