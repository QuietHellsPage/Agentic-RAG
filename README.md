# Agentic RAG

This is an Agentic RAG system that makes the process of finding specific information much easier. This RAG does not hallucinate. If Your question can not be answered using given context, Agent will say about it honestly.

## Features

1. Hybrid search: our retrieval search runs via both `dense` and `sparse` embeddings to combine the advantages od semantic and keywords search.

2. Using small chunks for searching and huge chunks for generating answer: Agent uses small chunks to find precise and accurate information, than uses huge chunks of these small chunks to answer the question. Thus, no context will be missed

3. Reformulating user question. We all can make irrelevant prompts, so, in case no information was found in the provided context, question will be reformulated to repeat the entire search pipeline.

**Important**

Remormulating feature can be used only once during single question-asnewring pipeline. This is made to avoid changing initial question too much.

## Quick Start Guide

### Converting pdf files to Markdown files

Initially, make sure that you have **Markdown** files. If You have not - use the module `src/helpers/process_raw_texts.py` to do this. To make this:

1. Add **.pdf** files to `data/raw_texts/pdf_storage`

2. In your IDE, run the file `src/helpers/process_raw_texts.py` or simply in terminal 

```bash 
python src/helpers/process_raw_texts.py
```
3. Your .md files are now stored in `data/raw_texts/md_storage`


### Installing LLMs

Now, you have to install Ollama app on Your PC to make LLMs run on it locally. To make this, go to the official Ollama site: https://ollama.com/

After installing Ollama, You need to install 2 models, needed for the RAG to work:

1. `mistral` is used inside of the Agent. Moreover, it is used to generate final answer. To install this model, go to the terminal and execute this command:

```bash
ollama pull mistral
```

**Important**
Do not forget to quit the model session at the end of working with RAG. Every time You will start it, in the separate terminal You will need to execute this command:

```bash
ollama serve
```
This command starts mistral session

2. `Qwen/Qwen3-Embedding-0.6B` `and Qdrant/bm25` are going to be installed aftomatically during first pipeline running, so there is no need to install them manually

### Running RAG locally

Now You are ready to run our Agentic RAG on Your PC. Follow the next steps:

1. Go to the terminal and execute

Linux/Mac OS
```bash
export PYTHONPATH=$(pwd)
```

Windows

```bash
do not know
```

2. Install all the needed libraries via

```bash
pip install -r requirements.txt
```

This process will take a while

3. Run the `src/interface/main.py`. You will see the progress bar of the information being loaded. It can take up to one hour if the text is pretty long.

4. Finally, you will see, that all the information is loaded. In your terminal You will see the link of the running `Gradio` server: `http://127.0.0.1:7860`. After following this link you will see the chat - answer Your question and get answers of them, or, if there is not enough information to answer the question - get honest response! To stop the Rag click `Ctrl + c` in the terminal. Good luck!