# Day 13 Performance Optimization Report

## Cache Result

- First request: 55.53 ms
- Repeated request: 0.03 ms
- Speedup: 1851.0x
- LLM calls: 1
- Cache stats: {'size': 1, 'hits': 1, 'misses': 1}

## Retrieval Parameter Comparison

| Top-K | Retrieval Top-K | Min similarity | Rerank | Accuracy | Elapsed ms |
| ---: | ---: | --- | --- | ---: | ---: |
| 2 | 2 | None | False | 100.00% | 13.54 |
| 3 | 5 | None | True | 100.00% | 13.41 |
| 3 | 5 | 0.05 | True | 100.00% | 14.96 |
| 4 | 5 | 0.05 | True | 100.00% | 18.23 |

## Recommended Configuration

- Top-K: 3
- Retrieval Top-K: 5
- Min similarity score: None
- Rerank enabled: True

## Notes

- In-memory TTL/LRU cache is enough for local development and single-process deployment.
- Redis can replace `InMemoryQACache` later when multiple backend workers share cache state.
- Prompt text was shortened to enforce concise context-only answers.
