import os
import sys
import argparse
import logging
from typing import List, Dict, Tuple
import warnings

# Suppress warnings for cleaner CLI
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" 
logging.getLogger("transformers").setLevel(logging.ERROR)

# Third-party libraries
try:
    import torch
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    import pdfplumber
    import docx
    from rich.console import Console
    from rich.table import Table
    from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn
    from rich.panel import Panel
    from rich.layout import Layout
    from rich.text import Text
    from rich import box
except ImportError as e:
    print(f"Missing dependency: {e.name}")
    print("Please run: pip install torch transformers pdfplumber python-docx rich")
    sys.exit(1)

# --- Configuration ---
MODEL_NAME = "Hello-SimpleAI/chatgpt-detector-roberta"
# Alternative models to try if you have more RAM:
# MODEL_NAME = "roberta-base-openai-detector" 

CHUNK_SIZE = 510  # Max tokens for Roberta is usually 512
CONFIDENCE_THRESHOLD = 0.5

console = Console()

class DocumentProcessor:
    """Handles file reading and text extraction."""
    
    @staticmethod
    def read_file(file_path: str) -> str:
        ext = os.path.splitext(file_path)[1].lower()
        if ext == '.pdf':
            return DocumentProcessor._read_pdf(file_path)
        elif ext == '.docx':
            return DocumentProcessor._read_docx(file_path)
        elif ext == '.txt':
            return DocumentProcessor._read_txt(file_path)
        else:
            raise ValueError(f"Unsupported file format: {ext}")

    @staticmethod
    def _read_pdf(file_path: str) -> str:
        text = ""
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n\n"
        return text

    @staticmethod
    def _read_docx(file_path: str) -> str:
        doc = docx.Document(file_path)
        return "\n\n".join([para.text for para in doc.paragraphs if para.text.strip()])

    @staticmethod
    def _read_txt(file_path: str) -> str:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()

class AIDetector:
    """Handles the AI detection logic using Hugging Face."""
    
    def __init__(self):
        self.device = 0 if torch.cuda.is_available() else -1
        console.print(f"[dim]Loading model: {MODEL_NAME} on {'GPU' if self.device == 0 else 'CPU'}...[/dim]")
        
        # Load Tokenizer and Model
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
        
        # Initialize Pipeline
        self.classifier = pipeline(
            "text-classification", 
            model=self.model, 
            tokenizer=self.tokenizer, 
            device=self.device,
            truncation=True,
            max_length=512,
            top_k=None # Get scores for all labels
        )

    def analyze_text(self, text: str) -> Dict:
        """
        Splits text into chunks and analyzes each.
        Returns detailed statistics.
        """
        # 1. Split text into paragraphs/chunks to fit model context
        # Simple splitting by newlines for logical separation, then ensuring generic length
        raw_chunks = [p for p in text.split('\n\n') if len(p.split()) > 10] # Ignore tiny fragments
        
        if not raw_chunks:
            return {"error": "No sufficient text found to analyze."}

        results = []
        ai_chunks = 0
        total_confidence = 0.0

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.percentage:>3.0f}%"),
            console=console
        ) as progress:
            task = progress.add_task("[cyan]Scanning document segments...", total=len(raw_chunks))
            
            for chunk in raw_chunks:
                # Truncate chunk purely for safety before passing to pipeline (pipeline handles truncation too)
                prediction = self.classifier(chunk[:2000]) # Pass first ~2000 chars to cover ~512 tokens
                
                # Parse prediction (Structure: [[{'label': 'ChatGPT', 'score': 0.9}, {'label': 'Human', 'score': 0.1}]])
                # Note: Output format depends on specific model version, normalizing below:
                scores = {item['label']: item['score'] for item in prediction[0]}
                
                # Normalize labels (Some models use 'Fake'/'Real', others 'ChatGPT'/'Human')
                ai_score = scores.get('ChatGPT', scores.get('Fake', scores.get('AI', 0.0)))
                human_score = scores.get('Human', scores.get('Real', 0.0))
                
                is_ai = ai_score > human_score
                
                if is_ai:
                    ai_chunks += 1
                
                results.append({
                    "text": chunk[:100] + "..." if len(chunk) > 100 else chunk,
                    "is_ai": is_ai,
                    "ai_confidence": ai_score,
                    "human_confidence": human_score
                })
                
                progress.update(task, advance=1)

        # Aggregation
        total_chunks = len(raw_chunks)
        ai_percentage = (ai_chunks / total_chunks) * 100 if total_chunks > 0 else 0
        
        return {
            "total_chunks": total_chunks,
            "ai_chunks": ai_chunks,
            "ai_percentage": ai_percentage,
            "chunk_details": results
        }

def display_report(filename: str, stats: Dict):
    """Generates a rich CLI dashboard."""
    
    if "error" in stats:
        console.print(f"[bold red]Error:[/bold red] {stats['error']}")
        return

    # --- Header ---
    console.print(Panel.fit(
        f"[bold blue]AI Content Analysis Report[/bold blue]\n[white]File: {filename}[/white]",
        border_style="blue"
    ))

    # --- Summary Statistics ---
    ai_pct = stats['ai_percentage']
    color = "green" if ai_pct < 20 else "yellow" if ai_pct < 50 else "red"
    verdict = "Likely Human" if ai_pct < 20 else "Mixed / Edited" if ai_pct < 50 else "Likely AI / GPT Generated"

    grid = Table.grid(expand=True)
    grid.add_column(justify="center", ratio=1)
    grid.add_column(justify="center", ratio=1)
    
    grid.add_row(
        Panel(f"[bold {color} size=24]{ai_pct:.1f}%[/]", title="AI Probability Score", border_style=color),
        Panel(f"[bold {color} size=18]{verdict}[/]", title="Overall Verdict", border_style=color)
    )
    console.print(grid)
    console.print("\n")

    # --- Detailed Breakdown Table ---
    table = Table(title="Segment Analysis (Paragraph Level)", box=box.SIMPLE_HEAD)
    
    table.add_column("ID", justify="right", style="cyan", no_wrap=True)
    table.add_column("Snippet", style="white")
    table.add_column("Prediction", justify="center")
    table.add_column("Confidence", justify="right")

    for i, detail in enumerate(stats['chunk_details']):
        # Only show top 10 and bottom 5 if too long, or all if short
        if len(stats['chunk_details']) > 15 and 5 < i < len(stats['chunk_details']) - 5:
            if i == 6:
                table.add_row("...", "...", "...", "...")
            continue
            
        pred_label = "[bold red]AI / LLM[/]" if detail['is_ai'] else "[green]Human[/]"
        conf_val = detail['ai_confidence'] if detail['is_ai'] else detail['human_confidence']
        
        table.add_row(
            str(i+1),
            detail['text'].replace("\n", " "),
            pred_label,
            f"{conf_val:.2%}"
        )

    console.print(table)
    console.print(f"\n[dim italic]Analyzed {stats['total_chunks']} segments. Detects patterns common in GPT-3.5/4/Gemini.[/dim italic]")

def main():
    parser = argparse.ArgumentParser(description="Analyze PDF/Docx/Txt for AI Generated Content")
    parser.add_argument("file", help="Path to the document file")
    args = parser.parse_args()

    if not os.path.exists(args.file):
        console.print(f"[bold red]Error:[/bold red] File '{args.file}' not found.")
        return

    try:
        # 1. Read
        console.print(f"[bold green] Reading file...[/]")
        text_content = DocumentProcessor.read_file(args.file)
        
        if len(text_content.strip()) == 0:
            console.print("[bold red]Error:[/bold red] File appears empty or text could not be extracted.")
            return

        # 2. Initialize Detector
        detector = AIDetector()
        
        # 3. Analyze
        stats = detector.analyze_text(text_content)
        
        # 4. Report
        display_report(os.path.basename(args.file), stats)

    except Exception as e:
        console.print(f"[bold red]Critical Error:[/bold red] {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
