import os
import requests
from urllib.parse import urlparse
from typing import List, Optional
import numpy as np
from fastapi import Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import jwt, jwk
from jose.utils import base64url_decode


from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi import Depends, Header
from jose import jwt, JWTError

from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl
from supabase import create_client
import tiktoken
from sentence_transformers import SentenceTransformer
from langchain_docling.loader import DoclingLoader
from langchain_core.documents import Document

# ---------- ENV ----------
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not SUPABASE_URL or not SUPABASE_KEY or not GROQ_API_KEY:
    raise RuntimeError("Supabase or Groq credentials missing")

# ---------- CONFIG ----------
TABLE_NAME = "medknowledge"
TOP_K = 5
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
GROQ_BASE = "https://api.groq.com/openai/v1"

# ---------- CLIENTS ----------
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
tokenizer = tiktoken.get_encoding("cl100k_base")
SUPABASE_JWT_SECRET = os.getenv("SUPABASE_JWT_SECRET")
SUPABASE_JWT_ISSUER = os.getenv("SUPABASE_JWT_ISSUER")
SUPABASE_PROJECT_REF = os.getenv("SUPABASE_PROJECT_REF")
JWKS_URL = f"https://{SUPABASE_PROJECT_REF}.supabase.co/auth/v1/.well-known/jwks.json"


# ---------- APP ----------
app = FastAPI(title="Medical RAG API")

# ---------- CORS ----------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)


# ---------- PDF Ingest Models ----------
class IngestRequest(BaseModel):
    filepath: HttpUrl


class IngestResponse(BaseModel):
    status: str
    message: str
    processed_urls: List[str]


# ---------- Chat Models ----------
class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: Optional[str] = "gpt-4o-mini"
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.0


# ---------- HELPERS ----------
def get_filename(url: str) -> str:
    return urlparse(url).path.split("/")[-1] or "document.pdf"


if not SUPABASE_JWT_SECRET:
    raise RuntimeError("SUPABASE_JWT_SECRET missing")

security = HTTPBearer()


def verify_supabase_jwt(
    credentials: HTTPAuthorizationCredentials = Security(security),
) -> str:
    token = credentials.credentials
    print("Token received:", token)

    try:
        # Step 1: Fetch JWKS from Supabase
        jwks = requests.get(JWKS_URL).json()
        headers = jwt.get_unverified_header(token)
        kid = headers.get("kid")
        if not kid:
            raise HTTPException(status_code=401, detail="Invalid token header: no kid")

        # Step 2: Find the key with matching kid
        key_data = next((k for k in jwks["keys"] if k["kid"] == kid), None)
        if not key_data:
            raise HTTPException(
                status_code=401, detail="Public key not found for token"
            )

        # Step 3: Construct JWK public key
        public_key = jwk.construct(key_data)

        # Step 4: Split token to verify signature
        message, encoded_signature = token.rsplit(".", 1)
        decoded_signature = base64url_decode(encoded_signature.encode("utf-8"))

        if not public_key.verify(message.encode("utf-8"), decoded_signature):
            raise HTTPException(status_code=401, detail="Invalid token signature")

        # Step 5: Decode claims (without verifying signature again)
        claims = jwt.get_unverified_claims(token)

        # Optional: verify expiry
        import time

        if claims.get("exp") and time.time() > claims["exp"]:
            raise HTTPException(status_code=401, detail="Token has expired")

        print("JWT claims:", claims)
        return claims["sub"]

    except Exception as e:
        print("JWT verification failed:", str(e))
        raise HTTPException(status_code=401, detail="Invalid or expired token")


def call_groq(messages, model="llama-3.1-8b-instant", temperature=0):
    url = f"{GROQ_BASE}/chat/completions"

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }

    serialized_messages = [{"role": m.role, "content": m.content} for m in messages]

    payload = {
        "model": model,
        "messages": serialized_messages,
        "temperature": temperature,
    }

    response = requests.post(url, json=payload, headers=headers, timeout=60)

    if response.status_code != 200:
        raise HTTPException(
            status_code=500,
            detail=response.text,
        )

    return response.json()["choices"][0]["message"]["content"]


def chunk_text(text: str, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    tokens = tokenizer.encode(text)
    chunks = []
    for i in range(0, len(tokens), chunk_size - overlap):
        chunks.append(tokenizer.decode(tokens[i : i + chunk_size]))
    return chunks


def embed_query(text: str) -> list[float]:
    emb = embedding_model.encode([text], normalize_embeddings=True)[0]
    return emb.tolist() if isinstance(emb, np.ndarray) else emb


def embed_texts(texts: list[str]) -> list[list[float]]:
    embeddings = embedding_model.encode(
        texts, normalize_embeddings=True, show_progress_bar=False
    )
    return [e.tolist() if isinstance(e, np.ndarray) else e for e in embeddings]


def extract_section(chunk: str) -> str:
    first_line = chunk.strip().split("\n")[0]
    return first_line[:120]


def extract_page_no(doc: Document, fallback: int) -> int:
    """
    Extract real PDF page number from Docling metadata.
    Fallback = page_index + 1
    """
    dl_meta = doc.metadata.get("dl_meta", {})
    items = dl_meta.get("doc_items", [])

    for item in items:
        prov = item.get("prov", [])
        if prov and "page_no" in prov[0]:
            return prov[0]["page_no"]

    return fallback


def ingest_pdf_urls(pdf_urls: list[str], user_id: str):

    for url in pdf_urls:
        response = requests.get(url, timeout=60)
        response.raise_for_status()

        filename = get_filename(url)
        with open(filename, "wb") as f:
            f.write(response.content)

        loader = DoclingLoader(file_path=filename)
        docs = loader.load()

        for page_index, doc in enumerate(docs):
            text = doc.page_content.strip()
            if not text:
                continue

            page_no = extract_page_no(doc, page_index + 1)

            print("doc.metadata:", doc.metadata)

            chunks = chunk_text(text)
            embeddings_list = embed_texts(chunks)

            rows = []
            for chunk, emb in zip(chunks, embeddings_list):
                rows.append(
                    {
                        "content": chunk,
                        "embedding": emb,
                        "url": url,
                        "document_title": filename,
                        "page_no": page_no,
                        "section": extract_section(chunk),
                        "user_id": user_id,
                    }
                )

            if rows:
                supabase.table(TABLE_NAME).insert(rows).execute()


# ---------- INGEST ENDPOINT ----------
@app.post("/ingest", response_model=IngestResponse)
async def ingest_pdf(
    request: IngestRequest, user_id: str = Depends(verify_supabase_jwt)
):
    try:
        url = str(request.filepath)
        ingest_pdf_urls([url], user_id)

        return IngestResponse(
            status="success", message="PDF ingested successfully", processed_urls=[url]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------- CHAT ENDPOINT ----------
def retrieve_context(query: str, user_id: str, k: int = TOP_K) -> List[Document]:
    query_embedding = embed_query(query)
    rpc = supabase.rpc(
        "match_medknowledge",
        {
            "p_query_embedding": query_embedding,
            "p_match_count": TOP_K,
            "p_user_id": user_id,
        },
    ).execute()

    docs = []
    for row in rpc.data:
        docs.append(
            Document(
                page_content=row["content"],
                metadata={
                    "url": row["url"],
                    "page_no": row["page_no"],
                    "section": row["section"],
                    "document_title": row["document_title"],
                },
            )
        )
    return docs


def build_prompt(user_question: str, docs: List[Document]) -> str:
    context_blocks = [
        f"[{d.metadata['document_title']} | page {d.metadata['page_no']} | {d.metadata['section']}]\n{d.page_content}"
        for d in docs
    ]
    context = "\n\n".join(context_blocks)
    return f"""
You are a medical information assistant.
Use ONLY the provided context.
If the answer is not present, say you do not know.

Context:
{context}

Question:
{user_question}
"""


# @app.post("/v1/chat/completions")
# def chat_completions(req: ChatRequest):
#     user_message = next(m.content for m in reversed(req.messages) if m.role == "user")
#     docs = retrieve_context(user_message)
#     prompt = build_prompt(user_message, docs)

#     # Groq API call
#     response = groq.chat_completion(
#         model=req.model or "gpt-4o-mini",
#         messages=[{"role": "user", "content": prompt}],
#         temperature=req.temperature,
#     )

#     return {
#         "id": "chatcmpl-medrag",
#         "choices": [
#             {
#                 "index": 0,
#                 "message": {
#                     "role": "assistant",
#                     "content": response.choices[0].message.content,
#                 },
#                 "finish_reason": "stop",
#             }
#         ],
#     }


@app.post("/v1/chat/completions")
def chat(
    req: ChatRequest,
    user_id: str = Depends(verify_supabase_jwt),
):
    user_question = next(m.content for m in reversed(req.messages) if m.role == "user")
    docs = retrieve_context(user_question, user_id=user_id)

    prompt = build_prompt(user_question, docs)
    answer = call_groq(
        messages=[ChatMessage(role="user", content=prompt)],
        temperature=req.temperature,
    )

    # optional dedupe by (doc, page)
    seen = set()
    citations = []
    for d in docs:
        key = (d.metadata["document_title"], d.metadata["page_no"])
        if key not in seen:
            seen.add(key)
            citations.append(
                {
                    "document_title": d.metadata["document_title"],
                    "page_no": d.metadata["page_no"],
                    "section": d.metadata["section"],
                    "url": d.metadata["url"],
                }
            )

    return {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": answer,
                }
            }
        ],
        "citations": citations,
    }


# ---------- Optional root ----------
@app.get("/")
async def root():
    return {"message": "FastAPI RAG backend running"}


# ---------- Offline Run Common Knwoeldge injsesiton ----------
if __name__ == "__main__":
    pdf_urls = [
        "https://www.cdc.gov/diabetes/pdfs/prevent/On-your-way-to-preventing-type-2-diabetes.pdf"
    ]

    ingest_pdf_urls(pdf_urls, "common_knowledge")
