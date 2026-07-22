"""A.R.I.A. ICD-10 RAG retriever.

ChromaDB-backed retriever for ICD-10 diagnosis codes.  Extracted from
agent_graph.py into its own module so F9 can rewrite the internals
(embedder swap, full code DB) behind a stable interface.

Public interface (preserved from original):
  - ICD10Retriever singleton class
  - get_icd_retriever() accessor
  - .search(query, n_results=3) -> list[dict]
"""

from __future__ import annotations

import logging
from typing import Optional

import chromadb

from data_loader import load_icd10_codes

logger = logging.getLogger(__name__)


class ICD10Retriever:
    """RAG retriever for ICD-10 codes using ChromaDB."""

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

        # Create or get collection
        self.collection = self.client.get_or_create_collection(
            name="icd10_codes",
            metadata={"description": "ICD-10 medical codes for diagnosis"},
        )

        # Populate if empty
        if self.collection.count() == 0:
            self._populate_collection()

        self._initialized = True

    def _populate_collection(self) -> None:
        """Populate ChromaDB with ICD-10 codes."""
        codes = load_icd10_codes()
        if not codes:
            logger.warning("No ICD-10 codes found to populate")
            return

        documents: list[str] = []
        metadatas: list[dict] = []
        ids: list[str] = []

        for code_data in codes:
            # Create searchable document from description and keywords
            doc = f"{code_data['description']}. Keywords: {', '.join(code_data['keywords'])}"
            documents.append(doc)
            metadatas.append({
                "code": code_data["code"],
                "description": code_data["description"],
            })
            ids.append(code_data["code"])

        self.collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids,
        )
        logger.info(f"Populated ChromaDB with {len(codes)} ICD-10 codes")

    def search(self, query: str, n_results: int = 3) -> list[dict]:
        """Search for relevant ICD-10 codes."""
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
