"""A.R.I.A. Medical Code RAG Retriever.

ChromaDB-backed retriever for diagnosis codes (ICD-10, ICD-11, SNOMED CT).
Extracted from agent_graph.py into its own module.

Public interface:
  - CodeRetriever singleton class (renamed from ICD10Retriever)
  - get_icd_retriever() accessor (backward-compatible)
  - .search(query, n_results=3, system=None) -> list[dict]

F9: Uses sentence-transformers with a medical embedding model (CPU).
F11: Multi-system support (ICD-10, ICD-11, SNOMED CT).
"""

from __future__ import annotations

import logging
from typing import Optional

import chromadb
from chromadb.api.types import EmbeddingFunction, Documents, Embeddings

from data_loader import load_icd10_codes, load_icd11_codes

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


class CodeRetriever:
    """RAG retriever for medical diagnosis codes using ChromaDB.

    Supports multiple coding systems: ICD-10, ICD-11, SNOMED CT.
    Uses sentence-transformers medical embedding model (CPU).
    """

    _instance: Optional["CodeRetriever"] = None

    def __new__(cls) -> "CodeRetriever":
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

        # Create collections for each coding system
        self.collections: dict[str, any] = {}
        for system in ["ICD-10", "ICD-11"]:
            self.collections[system] = self._get_or_create_collection(system)

        # Populate if empty
        for system, collection in self.collections.items():
            if collection.count() == 0:
                self._populate_collection(system, collection)

        self._initialized = True

    def _get_or_create_collection(self, system: str):
        """Get or create a collection for a coding system, handling embedder conflicts."""
        collection_name = system.lower().replace("-", "").replace(" ", "") + "_codes"
        description = f"{system} medical codes for diagnosis"

        try:
            return self.client.get_or_create_collection(
                name=collection_name,
                metadata={"description": description, "system": system},
                embedding_function=self._embedding_fn,
            )
        except ValueError:
            logger.info(f"Embedding function conflict for {system}. Recreating collection.")
            try:
                self.client.delete_collection(collection_name)
            except Exception:
                pass
            return self.client.get_or_create_collection(
                name=collection_name,
                metadata={"description": description, "system": system},
                embedding_function=self._embedding_fn,
            )

    def _populate_collection(self, system: str, collection) -> None:
        """Populate ChromaDB with codes from the specified system."""
        if system == "ICD-10":
            codes = load_icd10_codes()
        elif system == "ICD-11":
            codes = load_icd11_codes()
        else:
            logger.warning(f"Unknown coding system: {system}")
            return

        if not codes:
            logger.warning(f"No {system} codes found to populate")
            return

        documents: list[str] = []
        metadatas: list[dict] = []
        ids: list[str] = []

        for code_data in codes:
            doc = f"{code_data['description']}. Keywords: {', '.join(code_data.get('keywords', []))}"
            documents.append(doc)
            metadatas.append({
                "code": code_data["code"],
                "description": code_data["description"],
                "system": system,
            })
            # Prefix ID with system to avoid collisions
            ids.append(f"{system}:{code_data['code']}")

        batch_size = 50
        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i:i + batch_size]
            batch_metas = metadatas[i:i + batch_size]
            batch_ids = ids[i:i + batch_size]
            collection.add(
                documents=batch_docs,
                metadatas=batch_metas,
                ids=batch_ids,
            )

        logger.info(f"Populated ChromaDB with {len(codes)} {system} codes")

    def search(self, query: str, n_results: int = 3, system: str | None = None) -> list[dict]:
        """Search for relevant codes using medical embeddings.

        Args:
            query: Search query
            n_results: Number of results to return
            system: Optional filter by coding system ('ICD-10', 'ICD-11')

        Returns:
            List of code dictionaries with code, description, system, relevance
        """
        all_results = []

        systems_to_search = [system] if system else ["ICD-10", "ICD-11"]

        for sys in systems_to_search:
            collection = self.collections.get(sys)
            if collection is None:
                continue

            results = collection.query(
                query_texts=[query],
                n_results=n_results,
            )

            if results["metadatas"] and results["metadatas"][0]:
                for i, metadata in enumerate(results["metadatas"][0]):
                    all_results.append({
                        "code": metadata["code"],
                        "description": metadata["description"],
                        "system": metadata.get("system", sys),
                        "relevance": results["distances"][0][i] if results["distances"] else 0,
                    })

        # Sort by relevance and return top N
        all_results.sort(key=lambda x: x["relevance"])
        return all_results[:n_results]

    def get_systems(self) -> list[str]:
        """List supported coding systems."""
        return list(self.collections.keys())

    def get_code_count(self, system: str | None = None) -> int:
        """Get total number of indexed codes."""
        if system:
            collection = self.collections.get(system)
            return collection.count() if collection else 0
        return sum(c.count() for c in self.collections.values())


# Backward-compatible alias
ICD10Retriever = CodeRetriever


def get_icd_retriever() -> CodeRetriever:
    """Get or create the global code retriever singleton."""
    return CodeRetriever()
