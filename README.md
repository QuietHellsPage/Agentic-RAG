# Agentic RAG

An Agentic RAG system that makes finding specific information much easier.

> **This RAG does not hallucinate.** If your question cannot be answered using the provided context, the Agent will say so honestly.

---

## Features

### 1. Hybrid Search
Retrieval runs through both `dense` and `sparse` embeddings, combining the advantages of **semantic** and **keyword** search.

### 2. Small Chunks for Search, Large Chunks for Generation
The Agent uses small chunks to find precise and accurate information, then expands them to larger parent chunks when generating the final answer. This way, no important context is missed.

### 3. Question Reformulation
Sometimes prompts can be vague or imprecise. If no relevant information is found in the provided context, the question is automatically reformulated and the entire search pipeline is repeated.

> **Important:** Reformulation can only happen **once** per question. This prevents the original question from drifting too far from its initial intent.

---

## Quick Start Guide

### Step 1 — Convert PDF Files to Markdown

The RAG system works with **Markdown (`.md`)** files. If you only have PDFs, convert them first.

1. Place your `.pdf` files into `data/raw_texts/pdf_storage`

2. Run the conversion script:
   ```bash
   python src/helpers/process_raw_texts.py
   ```

3. The resulting `.md` files will appear in `data/raw_texts/md_storage`

---

### Step 2 — Install Ollama and Required Models

The RAG uses locally running LLMs via **Ollama**. Download and install it from the official site:
👉 [https://ollama.com/](https://ollama.com/)

After installing Ollama, pull the model used by the Agent:

```bash
ollama pull mistral
```

> **`Qwen/Qwen3-Embedding-0.6B`** and **`Qdrant/bm25`** will be downloaded **automatically** on the first run — no manual installation needed.

---

### Step 3 — Start the Ollama Server

Before running the RAG, you need to start the Ollama server. Open a **separate terminal window** and run:

```bash
ollama serve
```

Keep this terminal open while working with the RAG. When you're done, you can stop it with `Ctrl + C`.

---

### Step 4 — Set Up the Project and Install Dependencies

Open a new terminal in the **root directory of the project** and set the Python path:

#### Linux / macOS
```bash
export PYTHONPATH=$(pwd)
```

#### Windows (Command Prompt)
```cmd
set PYTHONPATH=%cd%
```

#### Windows (PowerShell)
```powershell
$env:PYTHONPATH = (Get-Location).Path
```

Then install all required libraries:

```bash
pip install -r requirements.txt
```

> This may take a few minutes depending on your internet connection.

---

### Step 5 — Configure the Device (Optional)

The device determines what hardware is used to compute embeddings. By default, **CPU** is used, so you can skip this step if you're unsure or just getting started.

However, if you want to speed up the indexing process, you can configure a faster device by creating a `.env` file in the **root directory of the project** with the following content:
```properties
EMBEDDINGS_DEVICE=
```

Set the value based on your hardware:

| Value | When to use |
|-------|-------------|
| `cpu` | Default — works on any machine |
| `cuda` | NVIDIA GPU |
| `mps` | Mac with Apple Silicon — use `cpu` instead if you run into memory issues

For example, on a Mac with Apple Silicon:
```properties
EMBEDDINGS_DEVICE=mps
```

---

### Step 6 — Run the RAG

Start the main interface:

```bash
python src/interface/main.py
```

You will see a progress bar as the system loads and indexes your documents. **This can take up to an hour** for large text collections — please be patient.

Once loading is complete, a link to the local **Gradio** interface will appear in the terminal:

```
http://127.0.0.1:7860
```

Open this link in your browser — you'll see a chat interface where you can ask questions and get answers based on your documents. If there isn't enough information to answer a question, the system will tell you honestly.

To stop the RAG, press **`Ctrl + C`** in the terminal.

---

Good luck! 🚀
