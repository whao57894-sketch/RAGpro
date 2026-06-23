import os
from pathlib import Path

from dotenv import load_dotenv
from zhipuai import ZhipuAI


def main() -> None:
    load_dotenv()
    for proxy_name in ["HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"]:
        os.environ.pop(proxy_name, None)

    api_key = os.getenv("ZHIPUAI_API_KEY")
    model = os.getenv("ZHIPUAI_MODEL", "glm-4-flash")

    if not api_key or api_key == "your_zhipu_api_key_here":
        raise SystemExit("ZHIPUAI_API_KEY is missing in .env")

    client = ZhipuAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": "用一句话回复：企业文档问答系统为什么需要标注出处？",
            }
        ],
        temperature=0.1,
        max_tokens=80,
    )

    answer = response.choices[0].message.content
    output_path = Path("day2_docs/zhipu_api_result.txt")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(answer, encoding="utf-8")
    print("Zhipu API OK")
    print(answer)


if __name__ == "__main__":
    main()
