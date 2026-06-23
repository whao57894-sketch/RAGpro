from pathlib import Path
import os
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.embeddings import ZhipuEmbeddingModel


def main() -> None:
    for proxy_name in ["HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"]:
        os.environ.pop(proxy_name, None)

    model = ZhipuEmbeddingModel(model="embedding-2")
    vector = model.embed_query("企业文档问答系统需要检索并标注出处")
    if not vector:
        raise SystemExit("Empty embedding returned")

    output_path = Path("docs/day5/zhipu_embedding_result.txt")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        f"dimension={len(vector)}\nfirst_5={vector[:5]}\n",
        encoding="utf-8",
    )

    print("Zhipu embedding OK")
    print(f"Dimension: {len(vector)}")
    print(f"First 5 values: {vector[:5]}")


if __name__ == "__main__":
    main()
