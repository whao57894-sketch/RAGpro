def test_core_imports():
    import dotenv
    import fastapi
    import langchain
    import langchain_community
    import pydantic
    import zhipuai

    assert dotenv
    assert fastapi
    assert langchain
    assert langchain_community
    assert pydantic
    assert zhipuai


if __name__ == "__main__":
    test_core_imports()
    print("Core imports OK")
