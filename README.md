# Multimodal RAG

Ask questions about a PDF that mixes text, tables and charts. Everything runs locally with Ollama, except vector storage, which uses Pinecone.

The sample document is `NovaCore_Multimodal_Company_Report_2026.pdf`, a report about a fictional company.

## How it works

```
PDF
 ├─ text    → page text (PyMuPDF)
 ├─ tables  → markdown tables (PyMuPDF find_tables)
 └─ images  → text summary from a vision model (qwen2.5vl:7b)
        ↓
Embeddings (sentence-transformers/all-MiniLM-L6-v2, 384 dims)
        ↓
Pinecone (index: novacore-multimodal-rag, namespace: SAT21NOV)
        ↓
Question → top 6 chunks → qwen3:4b answers with page citations
```

Each chunk carries `page`, `modality` (`text` / `table` / `visual`) and `source` metadata. Image summaries also keep `image_path`, so the original visual can be shown.

Vector IDs are stable (e.g. `...-p3-table-1`), so running the script again overwrites existing records rather than duplicating them.

## Setup

1. Install [Ollama](https://ollama.com) and pull the models:

   ```bash
   ollama pull qwen3:4b
   ollama pull qwen2.5vl:7b
   ```

2. Install the Python dependencies:

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

3. Create a `.env` file with your Pinecone key:

   ```
   PINECONE_API_KEY=your-key-here
   ```

## Run

```bash
python main.py
```

This extracts the PDF, summarizes the images, uploads all chunks to Pinecone, then answers the questions in the `for question in [...]` list at the end of `main.py`.

Example output:

```
Q: how was the revenue in usd in Q1 2025
Retrieved: [(3, 'visual'), (4, 'table'), (3, 'table'), (3, 'text'), (2, 'table'), (4, 'visual')]
A: $21.8M [p.3]
```

## Known limitations

- **Chart values are approximate.** The 7B vision model reads values off the chart image, and some come out a few tenths of a million off. A larger vision model reads them more accurately.
- **Full re-run every time.** Each run re-extracts, re-summarizes and re-uploads everything. That's fine for a small PDF, but slow for large ones.
- **qwen3 reasoning.** qwen3 sometimes writes out its reasoning even with `reasoning=False`, so `ask()` drops everything before `</think>`.
