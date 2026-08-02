"""
ChargeFlow AI V2 — RAG Text Embedder (Dense Sentence Embeddings)
=================================================================
Transforms text passages and queries into 384-dimensional dense semantic vectors
using SentenceTransformers (model: 'all-MiniLM-L6-v2').

Design choices:
  - Model: 'all-MiniLM-L6-v2' (384 dimensions, light & fast CPU inference)
  - L2-normalized embeddings so that dot-product equals exact Cosine Similarity in [0.0, 1.0]
  - Exposes clean abstraction: fit_transform(texts) and transform(texts)
  - Pure dense semantic vector representation
"""

import os
import sys
import site
import ctypes
import warnings
import numpy as np
from typing import List, Optional

# Windows PyTorch DLL loading helper (preloads OpenMP & C10 DLLs for background processes)
if sys.platform == "win32":
    try:
        paths = site.getsitepackages() + [site.getusersitepackages()]
    except Exception:
        paths = []
    for path in paths:
        torch_lib = os.path.join(path, "torch", "lib")
        if os.path.exists(torch_lib):
            if torch_lib not in os.environ.get("PATH", ""):
                os.environ["PATH"] = torch_lib + os.pathsep + os.environ.get("PATH", "")
            try:
                os.add_dll_directory(torch_lib)
            except Exception:
                pass
            for dll_file in ["libiomp5md.dll", "c10.dll", "torch_cpu.dll", "torch_python.dll"]:
                dll_path = os.path.join(torch_lib, dll_file)
                if os.path.exists(dll_path):
                    try:
                        ctypes.CDLL(dll_path)
                    except Exception:
                        pass

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    try:
        import torch
    except Exception:
        pass
    from sentence_transformers import SentenceTransformer


DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"


class TextEmbedder:
    """
    Transforms text chunks into dense 384-dimensional L2-normalized embeddings.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME):
        self.model_name = model_name
        self._model: Optional[SentenceTransformer] = None
        self.dimension = 384

    def _get_model(self) -> SentenceTransformer:
        if self._model is None:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self._model = SentenceTransformer(self.model_name)
        return self._model

    def _normalize(self, vectors: np.ndarray) -> np.ndarray:
        """Applies L2 normalization along axis=1."""
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1e-12
        return vectors / norms

    def fit_transform(self, texts: List[str]) -> np.ndarray:
        """
        Encodes document texts into a dense (N x 384) L2-normalized numpy matrix.
        """
        if not texts:
            raise ValueError("Cannot embed an empty list of document texts.")

        model = self._get_model()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            embeddings = model.encode(
                texts,
                convert_to_numpy=True,
                show_progress_bar=False,
                normalize_embeddings=True,
            )

        return self._normalize(embeddings.astype(np.float32))

    def transform(self, texts: List[str]) -> np.ndarray:
        """
        Encodes query texts into a dense (M x 384) L2-normalized numpy matrix.
        """
        if not texts:
            raise ValueError("Cannot embed an empty list of query texts.")

        model = self._get_model()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            embeddings = model.encode(
                texts,
                convert_to_numpy=True,
                show_progress_bar=False,
                normalize_embeddings=True,
            )

        return self._normalize(embeddings.astype(np.float32))
