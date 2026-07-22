"""A.R.I.A. ICD-10 RAG retriever.

ChromaDB-backed retriever for ICD-10 diagnosis codes.  Extracted from
agent_graph.py into its own module so F9 can rewrite the internals
(embedder swap, full code DB) behind a stable interface.

Public interface (preserved from original):
  - ICD10Retriever singleton class
  - get_icd_retriever() accessor
  - .search(query, n_results=3) -> list[dict]

F9: Uses sentence-transformers with a medical embedding model (CPU).
    Embedding model runs on CPU to preserve GPU VRAM for Whisper/LLM.
"""

from __future__ import annotations

import logging
from typing import Optional

import chromadb
from chromadb.api.types import EmbeddingFunction, Documents, Embeddings

from data_loader import load_icd10_codes

logger = logging.getLogger(__name__)

# Medical embedding model name (CPU-friendly, 80MB)
# Can be swapped to SapBERT/BioLORD for better medical accuracy
_MEDICAL_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class MedicalEmbeddingFunction(EmbeddingFunction):
    """Custom embedding function using sentence-transformers.

    Runs on CPU to preserve GPU VRAM for Whisper/LLM.
    Model is loaded lazily on first use.
    """

    _model = None
    _model_name: str = _MEDICAL_EMBEDDING_MODEL

    def __init__(self, model_name: str | None = None) -> None:
        if model_name:
            self._model_name = model_name

    def __call__(self, input: Documents) -> Embeddings:
        if MedicalEmbeddingFunction._model is None:
            logger.info(f"Loading embedding model: {self._model_name}")
            try:
                from sentence_transformers import SentenceTransformer
                MedicalEmbeddingFunction._model = SentenceTransformer(
                    self._model_name,
                    device="cpu",
                )
                logger.info("Embedding model loaded successfully (CPU)")
            except Exception as e:
                logger.error(f"Failed to load embedding model: {e}")
                # Fallback: use ChromaDB default embeddings
                raise

        embeddings = MedicalEmbeddingFunction._model.encode(
            input,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return embeddings.tolist()


class ICD10Retriever:
    """RAG retriever for ICD-10 codes using ChromaDB with medical embeddings."""

    _instance: Optional["ICD10Retriever"] = None

    def __new__(cls) -> "ICD10Retriever":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return

        # Use persistent storage, CPU-based to save VRAM
        self.client = chromadb.PersistentClient(path="./chroma_db")

        # F9: Use medical embedding function
        self._embedding_fn = MedicalEmbeddingFunction()

        self.collection = self._get_or_create_collection()

        # F9: If collection was created with a different embedder, rebuild
        expected_count = len(load_icd10_codes())
        current_count = self.collection.count()
        if current_count > 0 and current_count != expected_count:
            logger.info(
                f"Re-indexing: collection has {current_count} codes, "
                f"expected {expected_count}. Deleting and rebuilding."
            )
            self.client.delete_collection("icd10_codes")
            self.collection = self._get_or_create_collection()

        # Populate if empty
        if self.collection.count() == 0:
            self._populate_collection()

        self._initialized = True

    def _get_or_create_collection(self):
        """Get or create the ICD-10 collection, handling embedder conflicts.

        If the persisted collection was created with a different embedding
        function, ChromaDB raises ValueError.  We catch it, delete the
        stale collection, and create a fresh one with our MedicalEmbeddingFunction.
        """
        try:
            return self.client.get_or_create_collection(
                name="icd10_codes",
                metadata={"description": "ICD-10 medical codes for diagnosis"},
                embedding_function=self._embedding_fn,
            )
        except ValueError:
            # Embedding function conflict — stale collection from old embedder
            logger.info(
                "Embedding function conflict detected. "
                "Deleting stale collection and recreating with medical embeddings."
            )
            try:
                self.client.delete_collection("icd10_codes")
            except Exception:
                pass
            return self.client.get_or_create_collection(
                name="icd10_codes",
                metadata={"description": "ICD-10 medical codes for diagnosis"},
                embedding_function=self._embedding_fn,
            )

    def _populate_collection(self) -> None:
        """Populate ChromaDB with ICD-10 codes from the full dataset."""
        codes = load_icd10_codes()
        if not codes:
            logger.warning("No ICD-10 codes found to populate")
            return

        documents: list[str] = []
        metadatas: list[dict] = []
        ids: list[str] = []

        for code_data in codes:
            # Create searchable document from description and keywords
            doc = f"{code_data['description']}. Keywords: {', '.join(code_data.get('keywords', []))}"
            documents.append(doc)
            metadatas.append({
                "code": code_data["code"],
                "description": code_data["description"],
            })
            ids.append(code_data["code"])

        # Add in batches to avoid memory issues
        batch_size = 50
        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i:i + batch_size]
            batch_metas = metadatas[i:i + batch_size]
            batch_ids = ids[i:i + batch_size]
            self.collection.add(
                documents=batch_docs,
                metadatas=batch_metas,
                ids=batch_ids,
            )

        logger.info(f"Populated ChromaDB with {len(codes)} ICD-10 codes (medical embeddings)")

    def search(self, query: str, n_results: int = 3) -> list[dict]:
        """Search for relevant ICD-10 codes using medical embeddings."""
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
        )

        codes: list[dict] = []
        if results["metadatas"] and results["metadatas"][0]:
            for i, metadata in enumerate(results["metadatas"][0]):
                codes.append({
                    "code": metadata["code"],
                    "description": metadata["description"],
                    "relevance": results["distances"][0][i] if results["distances"] else 0,
                })

        return codes


def get_icd_retriever() -> ICD10Retriever:
    """Get or create the global ICD-10 retriever singleton."""
    return ICD10Retriever()
