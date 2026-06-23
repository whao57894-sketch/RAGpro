# Day 7 检索效果对比

| Mode | Total | Hits | Accuracy |
| --- | --- | --- | --- |
| Pure Vector | 10 | 8 | 0.8 |
| BM25 Only | 10 | 10 | 1.0 |
| Hybrid | 10 | 10 | 1.0 |

## Sample Queries

### Why must RAG answers cite source file names?

- Expected: answer_policy.txt
- Vector: answer_policy.txt, fallback_policy.txt, separator_policy.txt
- Hybrid: answer_policy.txt, fallback_policy.txt, metadata_policy.txt

### What does the vector database store and return?

- Expected: vector_policy.txt
- Vector: metadata_policy.txt, knowledge_base_policy.txt, vector_policy.txt
- Hybrid: vector_policy.txt, metadata_policy.txt, knowledge_base_policy.txt

### What should the system say when documents have no relevant information?

- Expected: fallback_policy.txt
- Vector: vector_policy.txt, fallback_policy.txt, upload_policy.txt
- Hybrid: fallback_policy.txt, vector_policy.txt, upload_policy.txt

### Why should text chunks not be too short?

- Expected: split_policy.txt
- Vector: split_policy.txt, embedding_policy.txt, upload_policy.txt
- Hybrid: split_policy.txt, embedding_policy.txt, upload_policy.txt

### Which separators should Chinese document splitting prefer?

- Expected: separator_policy.txt
- Vector: separator_policy.txt, vector_policy.txt, fallback_policy.txt
- Hybrid: separator_policy.txt, vector_policy.txt, fallback_policy.txt

### What does the enterprise knowledge base manage?

- Expected: knowledge_base_policy.txt
- Vector: knowledge_base_policy.txt, metadata_policy.txt, split_policy.txt
- Hybrid: knowledge_base_policy.txt, metadata_policy.txt, vector_policy.txt

### Which file formats can employees upload?

- Expected: upload_policy.txt
- Vector: hybrid_policy.txt, answer_policy.txt, separator_policy.txt
- Hybrid: hybrid_policy.txt, answer_policy.txt, upload_policy.txt

### Which metadata fields should parsed documents keep?

- Expected: metadata_policy.txt
- Vector: split_policy.txt, embedding_policy.txt, metadata_policy.txt
- Hybrid: split_policy.txt, metadata_policy.txt, embedding_policy.txt

### What is embedding-2 used for?

- Expected: embedding_policy.txt
- Vector: knowledge_base_policy.txt, separator_policy.txt, split_policy.txt
- Hybrid: knowledge_base_policy.txt, embedding_policy.txt, separator_policy.txt

### Why combine BM25 keyword search with vector search?

- Expected: hybrid_policy.txt
- Vector: metadata_policy.txt, hybrid_policy.txt, knowledge_base_policy.txt
- Hybrid: hybrid_policy.txt, metadata_policy.txt, knowledge_base_policy.txt
