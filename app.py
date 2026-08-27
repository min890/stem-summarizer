import streamlit as st
from pypdf import PdfReader
from groq import Groq

st.set_page_config(
    page_title="STEM Paper Summarizer",
    page_icon="🧪",
    layout="wide"
)

st.title("🧪 STEM Research Paper Summarizer")
st.caption("Upload a research paper (PDF) to extract clean math equations, methodology, and complete performance metrics.")

with st.sidebar:
    st.header("Settings")
    api_key = st.text_input("Groq API Key", type="password", help="Enter your gsk_... key here")
    st.markdown("---")
    st.markdown("Developed with Streamlit & Groq API")

def extract_pdf_text(uploaded_file) -> str:
    reader = PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted + "\n"
    return text

def analyze_paper(paper_text: str, key: str) -> str:
    client = Groq(api_key=key)

    prompt = f"""
    You are an expert STEM research assistant. Analyze the provided research paper text thoroughly.

    STRICT FORMATTING LAWS FOR MATHEMATICS:
    - NEVER use \\boxed{{}}, \\tag{{}}, \\displaystyle, \\bigl, \\Bigr, or square brackets \\[ \\].
    - Display equations MUST use double dollar signs on their own line:
      $$ equation $$
    - Inline symbols/variables MUST use single dollar signs:
      $\\eta$, $\\Delta m$, $P_s$, etc.
    - Write math symbols standardly (e.g., use \\text{{out}} instead of raw code blocks).

    STRUCTURE YOUR OUTPUT WITH THESE HEADERS:

    ## 🎯 1. Core Objective & Hypothesis
    Provide a concise summary of the core objective, problem solved, and primary hypothesis.

    ## 📐 2. Key Mathematical Equations & Variable Definitions
    List the key equations cleanly using $$...$$ syntax. 
    Below each equation, define variables cleanly using inline math $symbol$ = definition.

    ## ⚙️ 3. Experimental Setup & Methodology
    Outline the step-by-step procedure and materials clearly. Keep tables concise so they do not get cut off.

    ## 📊 4. Key Results & Performance Comparisons
    Provide a complete summary of data comparisons, key quantitative metrics, and experimental highlights. Ensure all bullet points and table rows are fully written out.

    ## ⚠️ 5. Limitations & Practical Applications
    Summarize constraints, trade-offs, and practical real-world applications.

    Paper Text:
    {paper_text[:18000]}
    """
    
    response = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama-3.3-70b-versatile",  # High token capacity model on Groq
        max_tokens=3500,
        temperature=0.2
    )
    return response.choices[0].message.content

uploaded_file = st.file_uploader("Drop your PDF research paper here", type=["pdf"])

if uploaded_file is not None:
    st.success(f"File loaded: **{uploaded_file.name}**")
    
    if st.button("🚀 Summarize Paper", type="primary"):
        if not api_key:
            st.error("Please enter your Groq API Key in the left sidebar first!")
        else:
            with st.spinner("Analyzing paper and formatting equations..."):
                try:
                    pdf_text = extract_pdf_text(uploaded_file)
                    summary = analyze_paper(pdf_text, api_key)
                    
                    st.markdown("---")
                    st.subheader("📋 Research Paper Summary")
                    st.markdown(summary)
                except Exception as e:
                    st.error(f"An error occurred: {e}")
