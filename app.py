import streamlit as st
from pypdf import PdfReader
from groq import Groq
import json

st.set_page_config(
    page_title="STEM Paper Summarizer & Lab Suite",
    page_icon="🧪",
    layout="wide"
)

# Sidebar Setup
with st.sidebar:
    st.header("⚙️ Settings")
    api_key = st.text_input("Groq API Key", type="password", help="Enter your gsk_... key here")
    st.markdown("---")
    st.markdown("### 💡 Quick Tips")
    st.markdown("- **Single Tab:** Full analysis, metrics, LaTeX copy & chat.")
    st.markdown("- **Compare Tab:** Side-by-side analysis of two papers.")

def extract_pdf_text(uploaded_file) -> str:
    reader = PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted + "\n"
    return text

def get_groq_client(key: str):
    return Groq(api_key=key)

def get_active_model(client):
    try:
        all_models = [m.id for m in client.models.list().data]
        # Priority list of full-sized text models (8k+ context windows)
        preferred = [
            "llama-3.3-70b-versatile", 
            "llama-3.1-8b-instant", 
            "llama3-70b-8192", 
            "llama3-8b-8192",
            "mixtral-8x7b-32768"
        ]
        for model in preferred:
            if model in all_models:
                return model
        # Fallback to first non-guard/non-whisper model
        valid = [m for m in all_models if "guard" not in m and "whisper" not in m and "safetensors" not in m]
        return valid[0] if valid else "llama-3.1-8b-instant"
    except Exception:
        return "llama-3.1-8b-instant"

def analyze_paper(paper_text: str, key: str) -> str:
    client = get_groq_client(key)
    chosen_model = get_active_model(client)

    prompt = f"""
    You are an expert STEM research assistant. Analyze the provided research paper text thoroughly.

    STRICT FORMATTING LAWS FOR MATHEMATICS:
    - NEVER use \\boxed{{}}, \\tag{{}}, \\displaystyle, \\bigl, \\Bigr, or square brackets \\[ \\].
    - Display equations MUST use double dollar signs on their own line:
      $$ equation $$
    - Inline symbols/variables MUST use single dollar signs:
      $\\eta$, $\\Delta m$, $P_s$, etc.

    STRUCTURE YOUR OUTPUT WITH THESE HEADERS:

    ## 🎯 1. Core Objective & Hypothesis
    Summary of problem and hypothesis.

    ## 📐 2. Key Mathematical Equations & Variable Definitions
    Equations in $$...$$ syntax and inline variable definitions.

    ## ⚙️ 3. Experimental Setup & Methodology
    Step-by-step procedure and materials.

    ## 📊 4. Key Results & Performance Comparisons
    Data comparisons and quantitative highlights.

    ## ⚠️ 5. Limitations & Practical Applications
    Constraints and real-world applications.

    Paper Text:
    {paper_text[:10000]}
    """
    
    response = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model=chosen_model,
        max_tokens=2000,
        temperature=0.2
    )
    return response.choices[0].message.content

def extract_metrics_and_glossary(paper_text: str, key: str):
    client = get_groq_client(key)
    chosen_model = get_active_model(client)

    prompt = f"""
    Analyze the text and output raw JSON ONLY with no markdown wrappers or formatting:
    {{
      "efficiency": "e.g., 76%",
      "evap_rate": "e.g., 0.22 kg m^-2 h^-1",
      "primary_material": "e.g., Carbon Black / Linen",
      "glossary": [
        {{"term": "AES", "definition": "Acoustically Enhanced System"}},
        {{"term": "FES", "definition": "Floating Evaporation System"}}
      ]
    }}

    Text excerpt:
    {paper_text[:6000]}
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
            "evap_rate": "N/A",
            "primary_material": "N/A",
            "glossary": []
        }

# Navigation Tabs
tab1, tab2 = st.tabs(["📄 Single Paper Analyzer", "⚖️ Compare Two Papers"])

# ==================== TAB 1: SINGLE PAPER ANALYZER ====================
with tab1:
    st.title("🧪 STEM Paper Deep-Dive & Lab Tools")
    uploaded_file = st.file_uploader("Upload a PDF research paper", type=["pdf"], key="single_pdf")

    if uploaded_file:
        st.success(f"Loaded: **{uploaded_file.name}**")
        
        if st.button("🚀 Run Full Analysis", type="primary", key="btn_single"):
            if not api_key:
                st.error("Please enter your Groq API Key in the left sidebar first!")
            else:
                with st.spinner("Analyzing text, parsing formulas, and extracting metrics..."):
                    pdf_text = extract_pdf_text(uploaded_file)
                    st.session_state['pdf_text'] = pdf_text
                    
                    summary = analyze_paper(pdf_text, api_key)
                    st.session_state['summary'] = summary
                    
                    metrics_data = extract_metrics_and_glossary(pdf_text, api_key)
                    st.session_state['metrics'] = metrics_data
                    st.session_state['chat_history'] = []

    if 'summary' in st.session_state:
        # Metrics Dashboard
        st.markdown("---")
        st.subheader("📊 Key Performance Metrics")
        m = st.session_state.get('metrics', {})
        col1, col2, col3 = st.columns(3)
        col1.metric("Peak Efficiency", m.get("efficiency", "N/A"))
        col2.metric("Evaporation Rate", m.get("evap_rate", "N/A"))
        col3.metric("Primary Material", m.get("primary_material", "N/A"))

        # Main Summary
        st.markdown("---")
        st.subheader("📋 Comprehensive Summary")
        st.markdown(st.session_state['summary'])

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
                        # Trim to 5000 chars per paper to stay safely below token limits
                        txt1 = extract_pdf_text(paper1)[:5000]
                        txt2 = extract_pdf_text(paper2)[:5000]

                        client = get_groq_client(api_key)
                        model = get_active_model(client)
                        
                        cmp_prompt = f"""
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
                            max_tokens=1500
                        )
                        st.markdown("---")
                        st.markdown(res.choices[0].message.content)
                    except Exception as e:
                        st.error(f"Comparison error: {e}")
