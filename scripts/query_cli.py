"""Interactive command-line interface for the complete local RAG pipeline."""
from pathlib import Path
from ragops.pipeline import RAGPipeline

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    pipeline = RAGPipeline(ROOT)
    print("RAGOps CLI. Type 'exit' to quit.")
    while True:
        question = input("\nQuestion: ").strip()
        if question.lower() in {"exit", "quit"}:
            break
        if not question:
            continue
        result = pipeline.query(question)
        print("\nAnswer:\n", result["answer"])
        print("\nSources:")
        for s in result["sources"]:
            print(f"- [{s['id']}] {s['title']} (rerank={s['rerank_score']})")
        print("\nMetrics:", result["metrics"])
        print("Latency:", result["latency_ms"])


if __name__ == "__main__":
    main()
