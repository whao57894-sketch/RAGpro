# Day 2 技术方案验证记录

## 验证目标

- 验证智谱 API 能否正常调用并返回结果。
- 准备测试 PDF，并验证 `pypdf` 能否正确提取文字。
- 验证 Chroma 向量数据库的写入和检索功能。
- 记录问题和解决方案，作为面试复盘素材。

## 已验证结果

- Python 虚拟环境可运行，核心依赖 `zhipuai`、`pypdf`、`langchain` 可导入。
- PDF 样例生成脚本可创建 `data/samples/enterprise_policy_test.pdf`。
- `pypdf` 可从测试 PDF 中提取文本，适合做 PDF 文档解析的第一版方案。
- 智谱 API 已完成真实调用，并返回正常回答，结果记录在 `day2_docs/zhipu_api_result.txt`。
- 切换到 Python 3.11.9 后，`chromadb==0.6.3` 可成功安装，向量写入和检索验证通过，结果记录在 `day2_docs/chroma_result.txt`。

## 遇到的问题

### pip target 安装后部分包目录损坏

现象：`pypdf`、`python-dotenv`、`zhipuai` 等包能被 Python 找到，但导入内部对象时报 `unknown location` 或 `Permission denied`。

原因：当前 Windows/Python 环境中，`pip --target` 创建的部分包目录权限异常，导致目录存在但不可正常读取。

解决方案：将必要 wheel 下载到 `vendor_wheels/`，用 Python `zipfile` 解压到 `local_packages/`，并通过 `.venv/Lib/site-packages/sitecustomize.py` 让 `local_packages/` 优先加载。

### 智谱 API 连接失败

现象：SDK 可导入，但 API 请求报 `WinError 10061`。

原因：环境变量中 `HTTP_PROXY`、`HTTPS_PROXY`、`ALL_PROXY` 指向 `http://127.0.0.1:9`，该端口拒绝连接。

解决方案：在 `scripts/day2/verify_zhipu_api.py` 中清理代理环境变量后再调用 SDK。验证通过。

### ChromaDB 安装失败

现象：安装 `chromadb==0.6.3` 时，依赖 `chroma-hnswlib==0.7.6` 需要本机编译，报错提示需要 Microsoft Visual C++ 14.0 或更高版本。

原因：当前机器使用 Python 3.13，`chroma-hnswlib` 在该环境下没有直接可用的预编译 wheel，pip 退回源码编译。

解决方案：

- 推荐方案：改用 Python 3.11 或 3.12 重新创建虚拟环境，RAG 生态兼容性更好。
- 备选方案：安装 Microsoft C++ Build Tools 后重试。
- 面试表达：这说明技术选型不仅要看框架能力，还要验证运行环境、Python 版本和底层 native 依赖是否匹配。

当前状态：已安装 Python 3.11.9 并创建 `.venv311`，ChromaDB 安装问题已解决。

### ChromaDB 默认 embedding 下载权限问题

现象：Chroma 默认 embedding 函数会尝试写入 `C:\Users\PC\.cache\chroma`，当前环境无权限。

解决方案：验证脚本中直接传入固定 `embeddings`，先验证 Chroma 的向量写入和检索能力。正式 RAG 链路会由项目自己的 embedding 模块生成向量，不依赖 Chroma 默认下载模型。

### ChromaDB 持久化 SQLite 写入问题

现象：`PersistentClient` 和直接使用 `sqlite3` 文件数据库时出现 `disk I/O error`。

解决方案：第 2 天先使用内存客户端完成写入/检索技术验证。后续进入知识库持久化阶段时，再单独处理 SQLite 落盘目录权限或改用可写数据盘路径。

## 面试素材

- 我没有直接假设向量数据库可用，而是写了最小写入和检索脚本做技术验证。
- PDF 解析选择 `pypdf` 作为第一版，是因为它轻量、安装简单，适合文本型 PDF；扫描件 OCR 会作为后续增强。
- 智谱 API 独立验证，避免后续把模型调用问题和检索链路问题混在一起排查。
- ChromaDB 暴露出 Python 3.13 兼容性问题，因此项目建议锁定 Python 3.11/3.12。
