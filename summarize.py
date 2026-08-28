import sys
import os
from pypdf import PdfReader
from groq import Groq
from rich.console import Console
from rich.markdown import Markdown

console = Console()

def get_groq_client():
    # Uses the API key provided
    return Groq

def extract_pdf_text(pdf_path: str) -> str:
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted + "\n"
    return text

def analyze_paper(paper_text: str) -> str:
    client = get_groq_client()
    
    # Auto-detect available models from Groq account
    try:
        available_models = [m.id for m in client.models.list().data]
        console.print(f"[dim]Available Groq models on your key: {', '.join(available_models[:4])}...[/dim]\n")
        
        # Priority list of preferred models
        preferred = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "llama3-8b-8192", "llama3-70b-8192"]
        chosen_model = next((m for m in preferred if m in available_models), available_models[0] if available_models else "llama-3.1-8b-instant")
    except Exception:
        chosen_model = "llama-3.1-8b-instant"

    console.print(f"[bold cyan]Using model:[/bold cyan] {chosen_model}")

    prompt = f"""
    You are an expert STEM research assistant. Analyze the following research paper text.
    
    Extract and format the output in Markdown with the following headers:
    ## 1. Core Objective & Hypothesis
    ## 2. Key Mathematical Equations (Render in LaTeX)
    ## 3. Methodology & Architecture
    ## 4. Key Results & Datasets
    ## 5. Limitations & Future Work

    Paper Text:
    {paper_text[:20000]}
    """
    
    response = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model=chosen_model,
    )
    return response.choices[0].message.content

if __name__ == "__main__":
    if len(sys.argv) < 2:
        console.print("[bold red]Usage:[/bold red] python summarize.py <path_to_pdf>")
        sys.exit(1)
        
    file_path = sys.argv[1]
    console.print(f"[bold blue]Extracting text from:[/bold blue] {file_path}...")
    
    try:
        text = extract_pdf_text(file_path)
        console.print("[bold green]Analyzing paper with Groq...[/bold green]\n")
        summary = analyze_paper(text)
        console.print(Markdown(summary))
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")