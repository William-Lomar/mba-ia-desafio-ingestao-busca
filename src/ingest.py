import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

from sqlalchemy import create_engine, inspect, text
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_postgres import PGVector

PDF_PATH = os.getenv("PDF_PATH")


def _already_ingested(database_url: str, collection_name: str, source: str) -> bool:
    engine = create_engine(database_url)
    try:
        if not inspect(engine).has_table("langchain_pg_embedding"):
            return False
        with engine.connect() as conn:
            result = conn.execute(
                text(
                    """
                    SELECT EXISTS (
                        SELECT 1
                        FROM langchain_pg_embedding e
                        JOIN langchain_pg_collection c ON c.uuid = e.collection_id
                        WHERE c.name = :collection_name
                          AND e.cmetadata->>'source' = :source
                    )
                    """
                ),
                {"collection_name": collection_name, "source": source},
            )
            return bool(result.scalar())
    finally:
        engine.dispose()


def ingest_pdf():
    for key in ("OPENAI_API_KEY", "DATABASE_URL", "PG_VECTOR_COLLECTION_NAME", "PDF_PATH"):
        if not os.getenv(key):
            raise RuntimeError(f"Environment variable {key} is not set")

    database_url = os.getenv("DATABASE_URL")
    collection_name = os.getenv("PG_VECTOR_COLLECTION_NAME")

    if _already_ingested(database_url, collection_name, PDF_PATH):
        print(f"{PDF_PATH} already ingested in collection '{collection_name}', skipping.")
        return

    print(f"Loading PDF from {PDF_PATH}")
    docs = PyPDFLoader(PDF_PATH).load()

    splits = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        add_start_index=False,
    ).split_documents(docs)

    print(f"Loaded {len(splits)} splits")

    if not splits:
        raise SystemExit(0)

    enriched = [
        Document(
            page_content=d.page_content,
            metadata={k: v for k, v in d.metadata.items() if v not in ("", None)},
        )
        for d in splits
    ]

    doc_id = Path(PDF_PATH).stem
    ids = [f"{doc_id}-{i}" for i in range(len(enriched))]

    embeddings = OpenAIEmbeddings(model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"))

    store = PGVector(
        embeddings=embeddings,
        collection_name=collection_name,
        connection=database_url,
        use_jsonb=True,
    )

    print("Ingesting to Postgres...")
    store.add_documents(documents=enriched, ids=ids)
    print("Done!")


if __name__ == "__main__":
    ingest_pdf()