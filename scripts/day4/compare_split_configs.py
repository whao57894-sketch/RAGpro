from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.document_parser import parse_document
from src.text_splitter import SplitConfig, compare_split_configs, split_documents


CONFIGS = [
    SplitConfig(chunk_size=250, chunk_overlap=40),
    SplitConfig(chunk_size=500, chunk_overlap=80),
    SplitConfig(chunk_size=800, chunk_overlap=120),
]


def main() -> None:
    sample_path = Path("data/day4_samples/long_policy_cn.txt")
    documents = parse_document(sample_path)
    rows = compare_split_configs(documents, CONFIGS)
    for row in rows:
        chunks = split_documents(
            documents,
            chunk_size=int(row["chunk_size"]),
            chunk_overlap=int(row["chunk_overlap"]),
        )
        row["retrieval_score"] = _estimate_retrieval_score(chunks)

    output_lines = [
        "# Day 4 文本切分参数对比",
        "",
        "| chunk_size | chunk_overlap | chunk_count | min_length | max_length | avg_length | quality_score | retrieval_score |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        output_lines.append(
            "| {chunk_size} | {chunk_overlap} | {chunk_count} | {min_length} | {max_length} | {avg_length} | {quality_score} | {retrieval_score} |".format(
                **row
            )
        )

    recommended = max(rows, key=lambda row: (row["retrieval_score"], row["quality_score"]))
    output_lines.extend(
        [
            "",
            "## 推荐参数",
            "",
            f"- chunk_size: {recommended['chunk_size']}",
            f"- chunk_overlap: {recommended['chunk_overlap']}",
            "",
            "选择理由：文本块数量适中，平均长度接近目标块大小，能保留较完整的中文语义片段，同时检索粒度不会过粗。",
            "",
            "## 样例切分结果",
            "",
        ]
    )

    chunks = split_documents(
        documents,
        chunk_size=int(recommended["chunk_size"]),
        chunk_overlap=int(recommended["chunk_overlap"]),
    )
    for chunk in chunks[:5]:
        output_lines.append(f"### Chunk {chunk.metadata['chunk_index']}")
        output_lines.append("")
        output_lines.append(chunk.page_content.strip())
        output_lines.append("")

    output_path = Path("docs/day4/split_comparison.md")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(output_lines), encoding="utf-8")

    print("\n".join(output_lines))
    print(f"\nSaved {output_path}")

def _estimate_retrieval_score(chunks) -> float:
    queries = ["大模型幻觉", "文件元数据", "chunk_overlap", "检索准确度"]
    score = 0
    for query in queries:
        matched = [chunk for chunk in chunks if query in chunk.page_content]
        if not matched:
            continue
        best_length = min(len(chunk.page_content) for chunk in matched)
        if 250 <= best_length <= 650:
            score += 25
        elif best_length < 250:
            score += 18
        else:
            score += 12
    return round(score, 2)


if __name__ == "__main__":
    main()
