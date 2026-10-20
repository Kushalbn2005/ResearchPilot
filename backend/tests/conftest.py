"""Shared test setup: a dummy DATABASE_URL, and a stub for sentence_transformers when it is not installed,
so the logic tests never load real models."""
import os
import sys
import types

os.environ.setdefault("DATABASE_URL", "postgresql://x:x@localhost/x")

try:
    import sentence_transformers  # noqa: F401
except ImportError:
    stub = types.ModuleType("sentence_transformers")
    stub.SentenceTransformer = object
    stub.CrossEncoder = object
    sys.modules["sentence_transformers"] = stub
