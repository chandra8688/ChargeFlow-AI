"""
ChargeFlow AI V2 — RAG Prompt Builder
======================================
Constructs strict, evidence-grounded prompt templates for LLM generation.

Design Principles:
  - Formats retrieved chunks in explicit XML evidence blocks `<evidence>`
  - Includes exact source filenames and section titles for citation tracing
  - Restricts LLM from using pre-trained external assumptions when answering
  - Directs model to explicitly state when evidence is insufficient
  - Uses scientifically accurate terminology (grounded response, hallucination-risk reduction)
"""

from typing import Dict, List, Any


SYSTEM_INSTRUCTION = """You are ChargeFlow AI Knowledge Assistant, a specialized AI assistant for the ChargeFlow EV Charging Orchestration Platform.

Your primary duty is to answer questions using ONLY the verified evidence snippets provided below.

RULES FOR ANSWERING:
1. Base your answer strictly on the provided evidence snippets inside <evidence> blocks.
2. Do NOT invent statistics, features, metrics, or technical mechanisms not present in the evidence.
3. Cite the relevant source document file names when stating key facts.
4. Keep terminology scientifically accurate (e.g. 'Mean Decrease in Impurity', 'occupancy rate', 'estimator dispersion').
5. If the evidence snippets do not contain enough information to answer the question accurately, explicitly state:
   "The available ChargeFlow AI knowledge base does not contain sufficient evidence to answer this question."
"""


class PromptBuilder:
    """
    Formats queries and retrieved evidence chunks into LLM prompts.
    """

    def build_prompt(self, query: str, retrieved_sources: List[Dict[str, Any]]) -> str:
        """
        Constructs the final prompt string from query and retrieved sources.
        """
        evidence_blocks = []
        for idx, src in enumerate(retrieved_sources, 1):
            block = (
                f'<snippet id="{idx}" source="{src["source"]}" section="{src.get("section_title", "")}">\n'
                f'{src["text"]}\n'
                f'</snippet>'
            )
            evidence_blocks.append(block)

        formatted_evidence = "\n\n".join(evidence_blocks)

        prompt = (
            f"<evidence>\n"
            f"{formatted_evidence}\n"
            f"</evidence>\n\n"
            f"QUESTION: {query}\n\n"
            f"Please provide a clear, factual answer grounded in the evidence snippets above."
        )

        return prompt

    def get_system_instruction(self) -> str:
        return SYSTEM_INSTRUCTION
