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
EMBEDDING_DIMENSION = 384  # must match EMBEDDING_MODEL's output size

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


# num_ctx: Ollama defaults to 4096 tokens; 3 images + retrieved context needs more.
vision_llm = ChatOllama(model=VISION_MODEL, temperature=0, num_predict=500, num_ctx=16384)


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

pc = Pinecone(
    api_key=os.environ["PINECONE_API_KEY"]  # fail fast with a clear KeyError if the key is missing
)

if not pc.has_index(PINECONE_INDEX_NAME):
    print("Creating Pinecone index...")

    # Pinecone SDK 7.x: create_index() blocks until the index is ready (timeout=None).
    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=EMBEDDING_DIMENSION,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1",
        ),
    )

    print("Pinecone index created.")
else:
    print("Pinecone index already exists.")

index = pc.Index(PINECONE_INDEX_NAME)

print("Connected to:", PINECONE_INDEX_NAME)

vector_store = PineconeVectorStore(
    index=index,
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


retriever = vector_store.as_retriever(search_kwargs={"k": 6})

# qwen3 can still emit its reasoning even with reasoning=False; keep only the final answer.
text_rag_chain = (
    ChatPromptTemplate.from_template(
        "Answer only from the context below. Mention page numbers like (Page 3). "
        "If the information is missing, say you could not find it.\n\n"
        "Context:\n{context}\n\nQuestion: {question}"
    )
    | text_llm
    | StrOutputParser()
    | (lambda text: text.split("</think>")[-1].strip())
)


def format_context(retrieved_docs):
    parts = []

    for doc in retrieved_docs:
        page = int(doc.metadata.get("page"))  # Pinecone returns numbers as floats (3.0)
        modality = doc.metadata.get("modality")

        parts.append(
            f"[Page {page} | {modality.upper()}]\n"
            f"{doc.page_content}"
        )

    return "\n\n".join(parts)


def answer_with_vision(question, context, image_paths):
    image_paths = image_paths[:3]

    content = [
        {
            "type": "text",
            "text": f"""
You are answering questions about the NovaCore Systems FY2026 report.

Use ONLY the retrieved context and the attached retrieved visuals.

RETRIEVED CONTEXT:
{context}

QUESTION:
{question}

Instructions:
- Answer factually.
- Use the attached visuals when relevant.
- Mention page numbers.
- If the information is missing, say you could not find it.
""",
        }
    ]

    for image_path in image_paths:
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": image_to_data_uri(image_path)
                },
            }
        )

    # Local Ollama vision model (no Groq account/key configured in this project).
    response = vision_llm.invoke([HumanMessage(content=content)])

    return response.content.strip()


def show_image(image_path):
    try:
        from IPython import get_ipython
        from IPython.display import display

        if get_ipython() is not None:  # running inside Jupyter
            display(Image.open(image_path))
            return
    except ImportError:
        pass
    Image.open(image_path).show()  # plain script: opens the OS image viewer


def ask_rag(question, show_sources=True, show_images=False):

    # 1. Retrieve from Pinecone
    retrieved_docs = retriever.invoke(question)

    # 2. Create context
    context = format_context(retrieved_docs)

    # 3. Check whether Pinecone returned visuals
    image_paths = []

    for doc in retrieved_docs:
        if doc.metadata.get("modality") == "visual":
            image_path = doc.metadata.get("image_path")

            if image_path and Path(image_path).exists():
                image_paths.append(image_path)

    # 4. Generate the answer
    if image_paths:
        answer = answer_with_vision(
            question=question,
            context=context,
            image_paths=image_paths,
        )
    else:
        answer = text_rag_chain.invoke(
            {
                "context": context,
                "question": question,
            }
        )

    print("\nQUESTION:")
    print(question)

    print("\nANSWER:")
    print(answer)

    if show_sources:
        print("\nPINECONE RETRIEVED SOURCES:")

        for i, doc in enumerate(retrieved_docs, start=1):
            print(
                f"{i}. Page {int(doc.metadata.get('page'))} "
                f"| {doc.metadata.get('modality')}"
            )

    if show_images and image_paths:
        print("\nRETRIEVED VISUALS:")

        for image_path in image_paths:
            show_image(image_path)

    return {
        "answer": answer,
        "documents": retrieved_docs,
        "image_paths": image_paths,
    }


time.sleep(10)  # serverless Pinecone: let the fresh upserts become searchable

for question in [
    "how was the revenue in usd in Q1 2025",
    "Where is the headquarters",
]:
    ask_rag(question)


