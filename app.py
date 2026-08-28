import json
import streamlit as st
from pypdf import PdfReader
from groq import Groq
import json
import re
from gtts import gTTS
import io
def get_groq_client(key: str):
    return Groq(api_key=key)
def get_active_model(client):
    try:
        models = client.models.list()
        for model in models.data:
            if "llama" in model.id.lower() or "mixtral" in model.id.lower():
                return model.id
        return models.data[0].id if models.data else "llama-3.3-70b-versatile"
    except Exception:
        return "llama-3.3-70b-versatile"
st.set_page_config(
    page_title="STEM Paper Summarizer & Lab Suite",
    page_icon="🧪",
    layout="wide"
)

# Sidebar Setup
with st.sidebar:
    st.header("⚙️ Settings & Persona")
    api_key = st.text_input("Groq API Key", type="password", help="Enter your gsk_... key here")
    
    persona = st.selectbox(
        "🎭 AI Explanation Persona",
        [
            "Standard STEM Expert",
            "Explain Like I'm an Undergrad",
            "Strict Math & Proofs Focus",
            "Commercial & Business Viability"
        ]
    )
    
    st.markdown("---")
    st.markdown("### 💡 Features Included")
    st.markdown("- **Single Paper Analyzer** (Summary, Metrics, Chat)")
    st.markdown("- **Compare Two Papers** (Side-by-side analysis)")
    st.markdown("- **Batch Summarizer** (Up to 5 papers into 1 matrix)")
    st.markdown("- **Paper Text Search** (Snippet matching)")
    st.markdown("- **Executive Audio Summary** (TTS generation)")

def extract_pdf_text(uploaded_file) -> str:
    reader = PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted + "\n"
    return text
def get_persona_prompt_modifier(persona_choice: str) -> str:
    persona_map = {
        "Standard STEM Expert": "Maintain strict scientific rigor and technical accuracy.",
        "Explain Like I'm an Undergrad": "Explain concepts clearly with intuitive analogies suitable for an undergraduate student.",
        "Strict Math & Proofs Focus": "Focus heavily on formal mathematical derivations, equations, and proofs.",
        "Commercial & Business Viability": "Analyze from a product/market lens, focusing on scalability and commercial potential."
    }
    return persona_map.get(persona_choice, "Maintain strict scientific rigor.")

def analyze_paper(paper_text, key, persona_choice, preset_focus=None):
    client = get_groq_client(key)
    chosen_model = get_active_model(client)

    persona_instruction = get_persona_prompt_modifier(persona_choice)

    focus_instruction = ""
    if preset_focus == "methodology":
        focus_instruction = "SPECIAL REQUEST: Focus predominantly on Section 3 (Experimental Setup & Methodology)."
    elif preset_focus == "data_tables":
        focus_instruction = "SPECIAL REQUEST: Focus predominantly on Section 4 (Key Results & Performance Data)."

    prompt = f"""
You are an expert STEM research assistant.
...
"""

    try:
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=chosen_model
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Groq API Error: {str(e)}")
        return "Error generating analysis."

def extract_metrics_and_glossary(paper_text: str, key: str):
    client = get_groq_client(key)
    chosen_model = get_active_model(client)
    
    prompt = f"""
    Analyze the text and output raw JSON ONLY with no markdown wrappers or formatting:
    {{
    "efficiency": "e.g., 76%",
    ...
    """
    
    try:
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=chosen_model,
            max_tokens=800,
            temperature=0.1
        )
        clean_json = response.choices[0].message.content.strip().replace("```json", "").replace("```", "")
        return json.loads(clean_json)
    except Exception:
        return {
            "efficiency": "N/A",
            "evap_rate": "N/A"
        }
        
    def generate_tts_audio(text: str) -> bytes:
        clean_text = re.sub(r'[*#$\\]', '', text)
        tts = gTTS(text=clean_text[:500], lang='en')
        audio_fp = io.BytesIO()
        tts.write_to_fp(audio_fp)
        audio_fp.seek(0)
        return audio_fp.read()

# Navigation Tabs
tab1, tab2, tab3 = st.tabs(["📄 Single Paper Analyzer", "⚖️ Compare Two Papers", "📂 Multi-Paper Batch Summarizer"])

# ==================== TAB 1: SINGLE PAPER ANALYZER ====================
with tab1:
    st.title("🧪 STEM Paper Deep-Dive & Lab Tools")
    uploaded_file = st.file_uploader("Upload a PDF research paper", type=["pdf"], key="single_pdf")

    preset = None
    col_p1, col_p2 = st.columns(2)
    with col_p1:
        if st.button("⚙️ Preset Focus: Methodology Only"):
            preset = "methodology"
    with col_p2:
        if st.button("📊 Preset Focus: Key Data & Tables"):
            preset = "data_tables"

    if uploaded_file:
        st.success(f"Loaded: **{uploaded_file.name}**")
        
        if st.button("🚀 Run Full Analysis", type="primary", key="btn_single") or preset:
            if not api_key:
                st.error("Please enter your Groq API Key in the left sidebar first!")
            else:
                with st.spinner(f"Analyzing text with persona '{persona}'..."):
                    pdf_text = extract_pdf_text(uploaded_file)
                    st.session_state['pdf_text'] = pdf_text
                    
                    summary = analyze_paper(pdf_text, api_key, persona, preset_focus=preset)
                    st.session_state['summary'] = summary
                    
    raw_json = extract_metrics_and_glossary(pdf_text, api_key)

                    # Strip out markdown backticks so json.loads can parse it properly
                if not raw_json or not isinstance(raw_json, str):
                        clean_json = "{}"
                    else:
                        clean_json = raw_json.replace("```json", "").replace("```", "").strip()

                    try:
                        metrics_data = json.loads(clean_json)
                    except Exception:
                        metrics_data = {}

                    st.session_state['metrics'] = metrics_data
                    st.session_state['chat_history'] = []

    if 'summary' in st.session_state:
        # TL;DR Executive Audio Section
        m = st.session_state.get('metrics', {})
        tldr_text = m.get("tldr", "No TLDR generated.")
        
        st.markdown("---")
        st.subheader("⚡ Executive Audio Briefing (TL;DR)")
        st.info(f"**TL;DR Summary:** {tldr_text}")
        if st.button("🔊 Generate Audio Summary"):
            with st.spinner("Generating audio snippet..."):
                audio_bytes = generate_tts_audio(tldr_text)
                st.audio(audio_bytes, format="audio/mp3")

# Metrics Dashboard
st.markdown("---")
st.subheader("📊 Key Performance Metrics & Data Table")

# 1. Standard Metric Cards (Already in your code)
col1, col2, col3 = st.columns(3)
col1.metric("Peak Efficiency", m.get("efficiency", "N/A"))
col2.metric("Evaporation Rate", m.get("evap_rate", "N/A"))
col3.metric("Primary Material", m.get("primary_material", "N/A"))

# 2. Add an interactive Data Table (NEW)
if m:
    # Filter out complex lists like 'glossary' to keep the table clean
    table_data = {k: v for k, v in m.items() if k != "glossary" and k != "tldr"}
    st.table(table_data)

# Main Summary
st.markdown("---")
st.subheader("📜 Comprehensive Summary")
st.markdown(st.session_state['summary'])
# Text Search & Highlight Tool
st.markdown("---")
st.subheader("🔍 Search Paper Snippets")
search_query = st.text_input("Enter a keyword to search within this paper (e.g., 'efficiency', 'laser', 'method'):")

if search_query and 'pdf_text' in st.session_state:
    lines = st.session_state['pdf_text'].splitlines()
    matches = [line for line in lines if search_query.lower() in line.lower()]
    if matches:
        st.success(f"Found {len(matches)} matching snippet(s):")
        for i, match in enumerate(matches[:5], 1):
            st.markdown(f"**{i}.** ...{match.strip()}...")
    else:
        st.warning("No direct snippet matches found.")

        # Interactive Glossary
        glossary = m.get("glossary", [])
        if glossary:
            with st.expander("📖 Technical Glossary & Acronyms"):
                for g in glossary:
                    st.markdown(f"**{g.get('term')}**: {g.get('definition')}")

        # Export & Download Options
        st.markdown("---")
        st.subheader("💾 Export & Copy Tools")
        ex_col1, ex_col2 = st.columns(2)
        
        with ex_col1:
            st.download_button(
                label="📥 Download Summary (.md)",
                data=st.session_state['summary'],
                file_name="paper_summary.md",
                mime="text/markdown"
            )
        
        with ex_col2:
            with st.expander("📋 Copy Extracted Equations"):
                equations = [line for line in st.session_state['summary'].split("\n") if "$$" in line or "\\eta" in line]
                eq_text = "\n".join(equations) if equations else "No stand-alone LaTeX blocks detected."
                st.code(eq_text, language="latex")

        # Interactive Chat with Paper
        st.markdown("---")
        st.subheader("💬 Chat with this Paper")
        
        for msg in st.session_state.get('chat_history', []):
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        user_query = st.chat_input("Ask a question about this paper's methods, dataset, or results...")
        if user_query:
            st.session_state['chat_history'].append({"role": "user", "content": user_query})
            with st.chat_message("user"):
                st.markdown(user_query)

            with st.chat_message("assistant"):
                with st.spinner("Searching paper context..."):
                    try:
                        client = get_groq_client(api_key)
                        model = get_active_model(client)
                        chat_prompt = f"Paper Context:\n{st.session_state['pdf_text'][:8000]}\n\nUser Question: {user_query}"
                        res = client.chat.completions.create(
                            messages=[{"role": "user", "content": chat_prompt}],
                            model=model,
                            max_tokens=600
                        )
                        ans = res.choices[0].message.content
                        st.markdown(ans)
                        st.session_state['chat_history'].append({"role": "assistant", "content": ans})
                    except Exception as e:
                        st.error(f"Error answering query: {e}")

# ==================== TAB 2: PAPER COMPARISON ====================
with tab2:
    st.title("⚖️ Side-by-Side Paper Comparison")
    st.caption("Upload two papers to generate a comparative breakdown of their methods, efficiency, and findings.")

    c1, c2 = st.columns(2)
    with c1:
        paper1 = st.file_uploader("Upload Paper A", type=["pdf"], key="cmp_pdf1")
    with c2:
        paper2 = st.file_uploader("Upload Paper B", type=["pdf"], key="cmp_pdf2")

    if paper1 and paper2:
        if st.button("⚖️ Compare Papers", type="primary"):
            if not api_key:
                st.error("Please enter your Groq API Key first!")
            else:
                with st.spinner("Extracting and comparing both papers..."):
                    try:
                        txt1 = extract_pdf_text(paper1)[:5000]
                        txt2 = extract_pdf_text(paper2)[:5000]

                        client = get_groq_client(api_key)
                        model = get_active_model(client)
                        persona_inst = get_persona_prompt_modifier(persona)
                        
                        cmp_prompt = f"""
                        {persona_inst}
                        Compare these two research papers in Markdown:

                        ### ⚔️ Direct Comparison Table
                        | Metric / Feature | Paper A ({paper1.name}) | Paper B ({paper2.name}) |
                        |---|---|---|
                        | Core Objective | ... | ... |
                        | Primary Material/Method | ... | ... |
                        | Efficiency / Performance | ... | ... |
                        | Key Limitation | ... | ... |

                        ### 📝 Key Differences & Synthesis
                        - Highlight 3 key technical differences.
                        - State which paper achieves better quantitative performance and why.

                        Paper A Text:
                        {txt1}

                        Paper B Text:
                        {txt2}
                        """

                        res = client.chat.completions.create(
                            messages=[{"role": "user", "content": cmp_prompt}],
                            model=model,
                            max_tokens=3000
                        )
                        st.markdown("---")
                        st.markdown(res.choices[0].message.content)
                    except Exception as e:
                        st.error(f"Comparison error: {e}")

# ==================== TAB 3: BATCH MULTI-PAPER SUMMARIZER ====================
with tab3:
    st.title("📂 Multi-Paper Batch Matrix Summarizer")
    st.caption("Upload up to 5 papers to extract a unified comparative summary table.")

    batch_files = st.file_uploader("Upload 2-5 PDF Research Papers", type=["pdf"], accept_multiple_files=True, key="batch_pdfs")

    if batch_files:
        if len(batch_files) > 5:
            st.warning("Please upload a maximum of 5 papers at once for optimal rate limits.")
        else:
            if st.button("🚀 Summarize All Papers into Matrix", type="primary"):
                if not api_key:
                    st.error("Please enter your Groq API Key first!")
                else:
                    with st.spinner(f"Processing {len(batch_files)} papers in batch..."):
                        batch_summaries = []
                        client = get_groq_client(api_key)
                        model = get_active_model(client)

                        for file in batch_files:
                            txt = extract_pdf_text(file)[:3500]
                            prompt = f"""
                            Provide a concise summary of this paper:
                            FileName: {file.name}
                            
                            Output JSON ONLY:
                            {{
                                "filename": "{file.name}",
                                "objective": "1-sentence summary of objective",
                                "methodology": "Primary method/materials used",
                                "key_result": "Main quantitative metric or finding"
                            }}

                            Text: {txt}
                            """
                            try:
                                res = client.chat.completions.create(
                                    messages=[{"role": "user", "content": prompt}],
                                    model=model,
                                    max_tokens=400,
                                    temperature=0.1
                                )
                                c_json = res.choices[0].message.content.strip().replace("```json", "").replace("```", "")
                                batch_summaries.append(json.loads(c_json))
                            except Exception:
                                batch_summaries.append({
                                    "filename": file.name,
                                    "objective": "Error processing",
                                    "methodology": "N/A",
                                    "key_result": "N/A"
                                })

                        st.markdown("---")
                        st.subheader("📊 Batch Summary Matrix")
                        
                        markdown_table = "| Paper Name | Core Objective | Methodology & Materials | Key Findings |\n|---|---|---|---|\n"
                        for item in batch_summaries:
                            markdown_table += f"| **{item.get('filename')}** | {item.get('objective')} | {item.get('methodology')} | {item.get('key_result')} |\n"
                        
                        st.markdown(markdown_table)
