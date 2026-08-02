"""
ChargeFlow AI V2 — RAG Document Loader
=======================================
Loads trustworthy internal documentation and metadata artifacts from the
repository to build the RAG knowledge corpus.

Whitelisted Sources:
  - README.md
  - docs/problem_framing.md
  - docs/demo_script.md
  - artifacts/models/demand_forecaster_metadata.json
  - artifacts/evaluation_report.json
  - artifacts/feature_importance.csv

Explicitly Excluded:
  - hourly_charging_data.csv (raw timeseries table)
  - logs/inference_log.jsonl (ephemeral runtime logs)
  - demand_forecaster.joblib (binary model file)
  - Raw source code files and caches
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

DEFAULT_KNOWLEDGE_FILES = [
    PROJECT_ROOT / "README.md",
    PROJECT_ROOT / "docs" / "problem_framing.md",
    PROJECT_ROOT / "docs" / "demo_script.md",
    PROJECT_ROOT / "artifacts" / "models" / "demand_forecaster_metadata.json",
    PROJECT_ROOT / "artifacts" / "evaluation_report.json",
    PROJECT_ROOT / "artifacts" / "feature_importance.csv",
]


class DocumentLoader:
    """
    Loads raw text and metadata from specified repository files.
    """

    def __init__(self, file_paths: Optional[List[Path]] = None):
        self.file_paths = file_paths or DEFAULT_KNOWLEDGE_FILES

    def load_documents(self) -> List[Dict[str, Any]]:
        """
        Reads specified files and returns a list of raw document dicts:
            [{"source": str, "content": str, "type": str}, ...]
        """
        documents = []
        for path in self.file_paths:
            path = Path(path)
            if not path.exists():
                continue

            rel_path = str(path.relative_to(PROJECT_ROOT)) if PROJECT_ROOT in path.parents or path == PROJECT_ROOT else str(path)

            if path.suffix == ".json":
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    formatted_content = f"JSON Artifact: {path.name}\n" + json.dumps(data, indent=2)
                    documents.append({
                        "source": rel_path,
                        "content": formatted_content,
                        "type": "json_artifact",
                    })
                except Exception as e:
                    print(f"Warning: Could not parse JSON {path}: {e}")
            elif path.suffix == ".csv":
                try:
                    content = path.read_text(encoding="utf-8")
                    documents.append({
                        "source": rel_path,
                        "content": f"CSV Feature Importance Artifact:\n{content}",
                        "type": "csv_artifact",
                    })
                except Exception as e:
                    print(f"Warning: Could not read CSV {path}: {e}")
            elif path.suffix in [".md", ".txt"]:
                try:
                    content = path.read_text(encoding="utf-8")
                    documents.append({
                        "source": rel_path,
                        "content": content,
                        "type": "markdown_doc",
                    })
                except Exception as e:
                    print(f"Warning: Could not read Markdown file {path}: {e}")

        return documents
