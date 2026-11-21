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
Question → top 6 chunks
        ├─ any image retrieved? → qwen2.5vl:7b answers, looking at the context + up to 3 original images
        └─ text/tables only     → qwen3:4b answers from the context
        ↓
Answer with page numbers + retrieved sources, tables and images
```

Each chunk carries `page`, `modality` (`text` / `table` / `visual`) and `source` metadata. Image summaries also keep `image_path`, so the original image can be sent to the vision model at question time and shown with the answer.

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

## Asking questions

Each question goes through `ask_rag()`:

```python
ask_rag("how was the revenue in usd in Q1 2025")

# Turn parts of the output off:
ask_rag(question, show_sources=False, show_tables=False, show_images=False)
```

It returns `{"answer", "documents", "image_paths"}` and prints:

- **ANSWER**: the model's answer, with page numbers.
- **PINECONE RETRIEVED SOURCES**: the 6 retrieved chunks, as page and modality.
- **RETRIEVED TABLES**: each retrieved table, as markdown.
- **RETRIEVED VISUALS**: the path of each image the vision model looked at (up to 3). In Jupyter the image shows inline; from a terminal it opens in the system image viewer.

Example output (abridged):

```
QUESTION:
how was the revenue in usd in Q1 2025

ANSWER:
The revenue in USD for Q1 2025 was $21.8M. This can be found on Page 3 of the report
under the "Quarterly Revenue Trend" line graph.

PINECONE RETRIEVED SOURCES:
1. Page 3 | visual
2. Page 3 | table
3. Page 4 | table
...

RETRIEVED TABLES:

[Page 3]
| Metric           | FY2025   | FY2026   | Change   |
|:-----------------|:---------|:---------|:---------|
| Revenue          | $97.5M   | $132.0M  | +35.4%   |
...

RETRIEVED VISUALS:
novacore_extracted_images/page_3_image_1.png
```

## Known limitations

- **Chart values are approximate.** The 7B vision model reads values off the chart image, and some come out a few tenths of a million off. A larger vision model reads them more accurately.
- **Full re-run every time.** Each run re-extracts, re-summarizes and re-uploads everything. That's fine for a small PDF, but slow for large ones.
- **The vision model answers most questions.** The top 6 results usually include at least one image, so qwen2.5vl answers rather than qwen3. That's slower, but it reads chart values straight from the image.
- **All retrieved tables and images are shown,** including ones the answer didn't use.
- **Vision context size.** Up to 3 images plus the retrieved text exceed Ollama's default 4,096-token window, so `vision_llm` sets `num_ctx=16384`.
- **qwen3 reasoning.** qwen3 sometimes writes out its reasoning even with `reasoning=False`, so `text_rag_chain` drops everything before `</think>`.
