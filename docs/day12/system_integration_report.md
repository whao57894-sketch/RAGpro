# Day 12 System Integration Test Report

## Scope

This report validates the complete offline API flow: upload documents, parse, split, store vectors, ask questions, return answers, and cite sources.

## Summary

- Uploaded documents: 8
- Evaluation questions: 20
- Retrieval accuracy: 95.00%
- Answer accuracy: 90.00%
- Average QA response time: 13.45 ms
- P95 QA response time: 19.34 ms
- Max QA response time: 23.28 ms
- Average upload time: 48.61 ms
- Performance requirement: PASS

## Uploaded Documents

| File | Chunks | Vectors | Upload ms |
| --- | ---: | ---: | ---: |
| leave_policy.txt | 1 | 1 | 13.24 |
| expense_policy.docx | 5 | 5 | 36.06 |
| product_manual.pdf | 1 | 1 | 220.04 |
| vpn_faq.txt | 1 | 1 | 9.81 |
| security_policy.docx | 5 | 5 | 49.4 |
| support_faq.pdf | 1 | 1 | 13.0 |
| onboarding_guide.txt | 1 | 1 | 10.12 |
| sla_policy.docx | 5 | 5 | 37.2 |

## Question Results

| # | Scenario | Expected source | Retrieval | Answer | Response ms |
| ---: | --- | --- | --- | --- | ---: |
| 1 | HR policy | leave_policy.txt | PASS | PASS | 9.06 |
| 2 | HR policy | leave_policy.txt | PASS | PASS | 7.49 |
| 3 | HR policy | leave_policy.txt | PASS | FAIL | 8.17 |
| 4 | Finance policy | expense_policy.docx | PASS | PASS | 23.28 |
| 5 | Finance policy | expense_policy.docx | PASS | PASS | 19.34 |
| 6 | Finance policy | expense_policy.docx | PASS | PASS | 18.48 |
| 7 | Product manual | product_manual.pdf | PASS | PASS | 8.14 |
| 8 | Product manual | product_manual.pdf | PASS | PASS | 7.8 |
| 9 | Product manual | product_manual.pdf | PASS | PASS | 18.43 |
| 10 | IT FAQ | vpn_faq.txt | PASS | PASS | 8.4 |
| 11 | IT FAQ | vpn_faq.txt | PASS | PASS | 18.56 |
| 12 | IT FAQ | vpn_faq.txt | PASS | PASS | 17.21 |
| 13 | Security policy | security_policy.docx | FAIL | FAIL | 17.13 |
| 14 | Security policy | security_policy.docx | PASS | PASS | 17.79 |
| 15 | Security policy | security_policy.docx | PASS | PASS | 19.06 |
| 16 | Support FAQ | support_faq.pdf | PASS | PASS | 7.36 |
| 17 | Support FAQ | support_faq.pdf | PASS | PASS | 17.73 |
| 18 | Onboarding | onboarding_guide.txt | PASS | PASS | 8.43 |
| 19 | SLA | sla_policy.docx | PASS | PASS | 8.62 |
| 20 | SLA | sla_policy.docx | PASS | PASS | 8.58 |

## Issues And Solutions

- External LLM calls were replaced with a deterministic local model for repeatable integration testing.
- The test corpus is ASCII English to avoid Windows console encoding problems during automated verification.
- Chroma telemetry warnings may appear in the console, but they do not affect test results.

## Conclusion

The core technical choices are feasible when retrieval and answer accuracy meet the thresholds in the automated test.
