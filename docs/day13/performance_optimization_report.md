# Day 13 Performance Optimization Report

## Cache Result

- First request: 53.05 ms
- Repeated request: 0.06 ms
- Speedup: 884.17x
- LLM calls: 1
- Cache stats: {'size': 1, 'hits': 1, 'misses': 1}

## Retrieval Parameter Comparison

| Top-K | Retrieval Top-K | Min similarity | Rerank | Accuracy | Elapsed ms |
| ---: | ---: | --- | --- | ---: | ---: |
| 2 | 2 | None | False | 100.00% | 14.61 |
| 3 | 5 | None | True | 100.00% | 13.09 |
| 3 | 5 | 0.05 | True | 100.00% | 20.63 |
| 4 | 5 | 0.05 | True | 100.00% | 25.77 |

## Recommended Configuration

- Top-K: 3
- Retrieval Top-K: 5
- Min similarity score: None
- Rerank enabled: True

## Notes

- In-memory TTL/LRU cache is enough for local development and single-process deployment.
- Redis can replace `InMemoryQACache` later when multiple backend workers share cache state.
- Prompt text was shortened to enforce concise context-only answers.
