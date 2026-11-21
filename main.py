import os
import io
import time
import base64
from pathlib import Path
from getpass import getpass

import fitz
import pandas as pd

from PIL import Image

from pinecone import Pinecone, ServerlessSpec

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore

PDF_PATH = Path("./NovaCore_Multimodal_Company_Report_2026.pdf")

TEXT_MODEL = "qwen3:4b"
VISION_MODEL = "qwen2.5vl:7b"  # local; gemma4:31b-cloud needs Ollama cloud credits
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

PINECONE_INDEX_NAME = "novacore-multimodal-rag"
PINECONE_NAMESPACE = "SAT21NOV"

IMAGE_DIR = Path("novacore_extracted_images")
IMAGE_DIR.mkdir(exist_ok=True)

print("PDF : ", PDF_PATH)
print("Pinecone index:", PINECONE_INDEX_NAME)
print("Pinecone namespace:", PINECONE_NAMESPACE)

# Minimal .env loader (KEY=VALUE lines); existing env vars win.
for line in Path(".env").read_text().splitlines() if Path(".env").exists() else []:
    if "=" in line and not line.lstrip().startswith("#"):
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

vision_llm = ChatOllama(model=VISION_MODEL, temperature=0, num_predict=500)


def image_to_data_uri(image_path):
    image_path = Path(image_path)

    with Image.open(image_path) as img:
        img = img.convert("RGB")
        img.thumbnail((1600,1600))
        buffer = io.BytesIO()
        img.save(buffer,format="JPEG",quality=85)

    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"

# Instead:

# Image / Chart
#      ↓
# Vision model
#      ↓
# Text summary
#      ↓
# Embedding
#      ↓
# Pinecone
# We still keep the original image path so the final answer can use the real visual.

def summarize_visual(image_path, page_number):
    image_data = image_to_data_uri(image_path)

    prompt = f"""
This visual was extracted from page {page_number} of the NovaCore FY2026 company report.

Describe the useful business information visible in the visual.

If it is a chart or graph:
- state the chart title and axis units
- list EVERY data point with its label, one per line, e.g. "Q1 2025: $21.8M"
- mention highest/lowest values
- mention the main trend

If it is a diagram:
- identify important components
- explain the flow or relationships

If it is a normal business image:
- describe the useful factual information

Keep the description concise and factual.
"""

    response = vision_llm.invoke([
        HumanMessage(content=[
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": image_data}},
        ])
    ])

    return response.content.strip()



# text_llm = ChatOllama()

def table_to_dataframe(page, table):
    # PyMuPDF's own cell text drops spaces in this PDF ("Europeanexpansion", "8.1" -> "81"),
    # so rebuild each cell from the page's words that sit mostly inside the cell box.
    words = page.get_text("words")

    def cell_text(cell):
        if cell is None:
            return ""
        box = fitz.Rect(cell)
        inside = [
            w for w in words
            if (fitz.Rect(w[:4]) & box).get_area() > 0.5 * fitz.Rect(w[:4]).get_area()
        ]
        return " ".join(w[4] for w in inside)

    rows = [[cell_text(c) for c in row.cells] for row in table.rows]
    header = table.header

    # External header = sits above the table box, so every row is data.
    if header.external:
        return pd.DataFrame(rows, columns=[cell_text(c) for c in header.cells])

    return pd.DataFrame(rows[1:], columns=rows[0])


def extract_multimodal_documents(pdf_path):
    pdf = fitz.open(str(pdf_path))

    documents = []
    extracted_xrefs = set()

    for page_index in range(len(pdf)):
        page = pdf[page_index]
        page_number = page_index + 1

        print(f"Processing page {page_number}...")

        # 1. TEXT
        text = page.get_text("text").strip()

        if text:
            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "page": page_number,
                        "modality": "text",
                        "source": pdf_path.name,
                    },
                )
            )

        # 2. TABLES
        try:
            tables = page.find_tables().tables

            for table_number, table in enumerate(tables, start=1):
                df = table_to_dataframe(page, table)

                if not df.empty:
                    documents.append(
                        Document(
                            page_content=df.to_markdown(index=False),
                            metadata={
                                "page": page_number,
                                "modality": "table",
                                "table_number": table_number,
                                "source": pdf_path.name,
                            },
                        )
                    )

        except Exception as error:
            print("Table extraction warning:", error)

        # 3. IMAGES / CHARTS / DIAGRAMS -> local vision model summary
        for image_number, image_info in enumerate(page.get_images(full=True), start=1):
            xref = image_info[0]

            # Do not process an identical embedded image twice.
            if xref in extracted_xrefs:
                continue
            extracted_xrefs.add(xref)

            image_data = pdf.extract_image(xref)
            image_path = IMAGE_DIR / f"page_{page_number}_image_{image_number}.{image_data['ext']}"
            image_path.write_bytes(image_data["image"])

            try:
                summary = summarize_visual(image_path=image_path, page_number=page_number)

                documents.append(
                    Document(
                        page_content=summary,
                        metadata={
                            "page": page_number,
                            "modality": "visual",
                            "image_path": str(image_path),
                            "source": pdf_path.name,
                        },
                    )
                )

            except Exception as error:
                print(f"Vision warning on page {page_number}:", error)

    pdf.close()
    return documents


documents = extract_multimodal_documents(PDF_PATH)

print()
print("Total LangChain documents:", len(documents))
print("By modality:", pd.Series([d.metadata["modality"] for d in documents]).value_counts().to_dict())
print("Table pages:", [d.metadata["page"] for d in documents if d.metadata["modality"] == "table"])


# 4. EMBED + STORE IN PINECONE
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

pc = Pinecone(api_key=PINECONE_API_KEY)
if not pc.has_index(PINECONE_INDEX_NAME):
    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=384,  # all-MiniLM-L6-v2 output size
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

vector_store = PineconeVectorStore(
    index=pc.Index(PINECONE_INDEX_NAME),
    embedding=embeddings,
    namespace=PINECONE_NAMESPACE,
)

# Stable ids, so re-running overwrites instead of duplicating.
ids = [
    f"{d.metadata['source']}-p{d.metadata['page']}-{d.metadata['modality']}-"
    f"{d.metadata.get('table_number') or Path(d.metadata.get('image_path', '')).stem}"
    for d in documents
]
assert len(set(ids)) == len(ids), "duplicate Pinecone ids"
vector_store.add_documents(documents, ids=ids)
print("Upserted", len(ids), "documents to Pinecone")


# 5. RETRIEVE + ANSWER
text_llm = ChatOllama(model=TEXT_MODEL, temperature=0, reasoning=False)


def ask(question, k=6):
    # Serverless Pinecone is eventually consistent; fresh upserts can take a few seconds to show up.
    hits = vector_store.similarity_search(question, k=k)
    context = "\n\n".join(
        f"[p.{int(d.metadata['page'])} {d.metadata['modality']}]\n{d.page_content}" for d in hits
    )
    answer = text_llm.invoke(
        "Answer only from the context below. Cite pages like [p.3].\n\n"
        f"Context:\n{context}\n\nQuestion: {question}"
    ).content
    # qwen3 can still emit its reasoning even with reasoning=False; keep only the final answer.
    return answer.split("</think>")[-1].strip(), hits


time.sleep(10)  # let the fresh upserts become searchable

for question in [
    "❯ how was the revenue in usd in Q1 2025",
]:
    answer, hits = ask(question)
    print("\nQ:", question)
    print("Retrieved:", [(int(d.metadata["page"]), d.metadata["modality"]) for d in hits])
    print("A:", answer)


