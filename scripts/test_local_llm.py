"""One-time smoke test for the local Qwen generator.

Run:
    python scripts/test_local_llm.py

The first run downloads the model from Hugging Face and caches it locally.
"""
from ragops.generation import LocalGenerator


def main() -> None:
    generator = LocalGenerator()
    result = generator.generate(
        "What is model drift?",
        [{"id": "doc_01", "title": "Model drift", "text": "Model drift is a change in the relationship between model inputs, outputs, or labels over time."}],
        max_new_tokens=100,
    )
    print(result["answer"])


if __name__ == "__main__":
    main()
