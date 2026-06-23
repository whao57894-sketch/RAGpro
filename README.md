# 企业智能文档问答系统

这是一个面向企业内部知识库场景的 RAG 文档问答系统。用户上传 PDF、DOCX、TXT 文档后，系统会自动解析、切分、向量化并写入 Chroma 知识库；员工可以用自然语言提问，系统从已上传文档中检索相关内容，调用大模型生成答案，并返回引用来源。

核心目标是让回答基于企业自己的文档，而不是依赖模型泛化知识，从而降低幻觉，保证答案可追溯、可核查。

## 功能特性

- 支持 PDF、DOCX、TXT 文档上传
- 自动解析文档并保留文件名、路径、分块编号等元数据
- 使用递归字符切分器进行中文友好的文本切分
- 使用智谱 `embedding-2` 或本地确定性 embedding 进行向量化
- 使用 Chroma 作为本地向量数据库
- 支持向量检索、BM25 关键词检索、混合检索和轻量 rerank
- 问答引擎支持 Prompt 约束、引用来源、无答案兜底提示
- 后端使用 FastAPI，前端使用 Streamlit
- 支持内存缓存，减少高频问题重复调用
- 提供完整单元测试、集成测试和性能测试脚本

## 项目结构

```text
RAG/
├── api/                    # FastAPI 后端服务
│   └── main.py
├── frontend/               # Streamlit 前端页面
│   └── streamlit_app.py
├── src/                    # RAG 核心模块
│   ├── document_parser.py  # PDF/DOCX/TXT 文档解析
│   ├── text_splitter.py    # 文本切分
│   ├── embeddings.py       # Embedding 模型封装
│   ├── vector_store.py     # Chroma 向量库封装
│   ├── retrieval.py        # BM25 / 混合检索
│   ├── qa_engine.py        # 问答引擎
│   └── llm.py              # 智谱 LLM 调用封装
├── data/
│   ├── uploads/            # 运行时上传文件，默认可清空
│   ├── vector_store/       # Chroma 本地向量库，默认可清空
│   └── *_samples/          # 测试样例文档
├── scripts/                # 每个阶段的验证脚本
├── tests/                  # 自动化测试
├── docs/                   # 阶段验证报告和结果
├── requirements.txt        # 完整依赖
├── requirements-lite.txt   # 轻量依赖
├── .env.example            # 环境变量模板
└── verify_environment.py   # 环境检查脚本
```

## 环境要求

- Windows PowerShell
- Python 3.11，推荐 `3.11.9`
- 智谱 AI API Key

项目当前使用的虚拟环境目录是 `.venv311`。如果你在新电脑上运行，可以重新创建同名虚拟环境。

## 快速启动

### 1. 创建并激活虚拟环境

```powershell
python -m venv .venv311
.\.venv311\Scripts\Activate.ps1
```

如果 PowerShell 不允许执行脚本，可以先运行：

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

### 2. 安装依赖

```powershell
.\.venv311\Scripts\python.exe -m pip install -r requirements.txt
```

如果只想先安装核心依赖：

```powershell
.\.venv311\Scripts\python.exe -m pip install -r requirements-lite.txt
```

### 3. 配置环境变量

复制 `.env.example` 为 `.env`：

```powershell
Copy-Item .env.example .env
```

然后编辑 `.env`：

```env
ZHIPUAI_API_KEY=你的智谱API密钥
ZHIPUAI_MODEL=glm-4-flash
EMBEDDING_MODEL=embedding-2
VECTOR_STORE_DIR=data/vector_store
UPLOAD_DIR=data/uploads
```

注意：`.env` 里包含真实密钥，不要提交到代码仓库。

### 4. 验证环境

```powershell
.\.venv311\Scripts\python.exe verify_environment.py
```

看到下面输出说明环境基本可用：

```text
Environment verification OK
```

## 运行后端 API

启动 FastAPI 服务：

```powershell
.\.venv311\Scripts\python.exe -m uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
```

启动后访问：

- 健康检查：http://127.0.0.1:8000/health
- API 文档：http://127.0.0.1:8000/docs

主要接口：

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| `GET` | `/health` | 服务健康检查 |
| `POST` | `/documents/upload` | 上传并入库 PDF/DOCX/TXT |
| `GET` | `/documents` | 查看已入库文档 |
| `POST` | `/qa/ask` | 提问并返回答案和来源 |
| `DELETE` | `/documents/clear` | 清空知识库 |

## 运行前端页面

后端启动后，另开一个 PowerShell 窗口运行：

```powershell
.\.venv311\Scripts\python.exe -m streamlit run frontend\streamlit_app.py
```

默认前端会连接：

```text
http://127.0.0.1:8000
```

如果后端地址不同，可以设置环境变量：

```powershell
$env:RAG_API_BASE_URL="http://127.0.0.1:8000"
.\.venv311\Scripts\python.exe -m streamlit run frontend\streamlit_app.py
```

## 使用流程

1. 启动后端 API。
2. 启动 Streamlit 前端。
3. 在左侧上传 PDF、DOCX 或 TXT 文档。
4. 等待文档解析、切分、向量化并入库。
5. 在聊天框中提问。
6. 查看回答和引用来源文件名。

## 测试命令

运行全部测试：

```powershell
.\.venv311\Scripts\python.exe -m pytest tests -q
```

运行核心模块测试：

```powershell
.\.venv311\Scripts\python.exe -m pytest tests\test_document_parser.py tests\test_text_splitter.py tests\test_retrieval.py tests\test_vector_store.py tests\test_qa_engine.py -q
```

运行 API 链路测试：

```powershell
.\.venv311\Scripts\python.exe -m pytest tests\test_api_upload.py tests\test_day11_api_flow.py -q
```

## 常用验证脚本

系统集成测试：

```powershell
.\.venv311\Scripts\python.exe scripts\day12\run_integration_test.py
```

性能优化测试：

```powershell
.\.venv311\Scripts\python.exe scripts\day13\benchmark_performance.py
```

代码质量和敏感信息检查：

```powershell
.\.venv311\Scripts\python.exe scripts\day15\code_quality_check.py
```

对应报告会生成在：

- `docs/day12/`
- `docs/day13/`
- `docs/day15/`

## 运行时数据说明

这些目录是运行过程中自动产生或更新的：

- `data/uploads/`
- `data/vector_store/`

如果想重置知识库，可以调用接口：

```powershell
Invoke-RestMethod -Method Delete http://127.0.0.1:8000/documents/clear
```

也可以在停止服务后清空这两个目录。

## 常见问题

### 1. 上传文档后提问提示知识库为空

先确认上传接口返回 `200`，再访问：

```text
http://127.0.0.1:8000/documents
```

如果文档列表为空，说明上传或解析失败。

### 2. 智谱 API Key 报错

检查 `.env` 是否存在，并确认：

```env
ZHIPUAI_API_KEY=真实可用的密钥
```

不要把 `.env.example` 当作真实配置文件使用。

### 3. Chroma telemetry 警告

测试时可能看到 Chroma telemetry 相关警告，一般不影响本地功能。

### 4. pytest cache 权限警告

如果看到 `.pytest_cache` 权限警告，不影响测试结果。项目已尽量避免依赖 pytest cache。

## 技术栈

- Python 3.11
- FastAPI
- Streamlit
- LangChain document loaders / text splitters
- Chroma
- 智谱 AI `glm-4-flash` / `embedding-2`
- Pytest
