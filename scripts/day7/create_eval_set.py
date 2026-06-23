from pathlib import Path
import json


CORPUS = {
    "answer_policy.txt": "RAG answers must cite source file names. Source citation helps employees verify every answer.",
    "vector_policy.txt": "The vector database stores chunk embeddings and returns Top-K similar chunks for semantic retrieval.",
    "fallback_policy.txt": "If uploaded documents do not contain relevant information, the system must say no related information was found.",
    "split_policy.txt": "Text chunks should not be too short because short chunks lose context. Very long chunks make retrieval less precise.",
    "separator_policy.txt": "Chinese document splitting should prefer paragraph breaks, new lines, periods, question marks, exclamation marks, semicolons and commas.",
    "knowledge_base_policy.txt": "The enterprise knowledge base manages policies, product manuals, FAQ files and project documents for natural language search.",
    "upload_policy.txt": "Employees can upload PDF, DOCX and TXT documents. The parser converts them into standard Document objects.",
    "metadata_policy.txt": "Parsed documents must keep metadata including file_name, file_path, file_extension, page and chunk_index.",
    "embedding_policy.txt": "The embedding-2 model converts text into vectors so similar text chunks can be found by vector search.",
    "hybrid_policy.txt": "Hybrid retrieval combines BM25 keyword search with vector search to improve recall for exact terms and semantic questions.",
}


EVAL_ITEMS = [
    {"question": "Why must RAG answers cite source file names?", "expected_file": "answer_policy.txt"},
    {"question": "What does the vector database store and return?", "expected_file": "vector_policy.txt"},
    {"question": "What should the system say when documents have no relevant information?", "expected_file": "fallback_policy.txt"},
    {"question": "Why should text chunks not be too short?", "expected_file": "split_policy.txt"},
    {"question": "Which separators should Chinese document splitting prefer?", "expected_file": "separator_policy.txt"},
    {"question": "What does the enterprise knowledge base manage?", "expected_file": "knowledge_base_policy.txt"},
    {"question": "Which file formats can employees upload?", "expected_file": "upload_policy.txt"},
    {"question": "Which metadata fields should parsed documents keep?", "expected_file": "metadata_policy.txt"},
    {"question": "What is embedding-2 used for?", "expected_file": "embedding_policy.txt"},
    {"question": "Why combine BM25 keyword search with vector search?", "expected_file": "hybrid_policy.txt"},
]


def main() -> None:
    corpus_dir = Path("data/day7_corpus")
    corpus_dir.mkdir(parents=True, exist_ok=True)
    for file_name, content in CORPUS.items():
        (corpus_dir / file_name).write_text(content, encoding="utf-8")

    output_path = Path("day7_docs/eval_set.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(EVAL_ITEMS, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {output_path}")
    print(f"Saved corpus in {corpus_dir}")


if __name__ == "__main__":
    main()
