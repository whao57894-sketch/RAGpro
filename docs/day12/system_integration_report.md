# Day 12 System Integration Test Report

## Scope

This report validates the complete offline API flow: upload documents, parse, split, store vectors, ask questions, return answers, and cite sources.

## Summary

- Uploaded documents: 8
- Evaluation questions: 20
- Retrieval accuracy: 90.00%
- Answer accuracy: 85.00%
- Average QA response time: 6.76 ms
- P95 QA response time: 7.66 ms
- Max QA response time: 7.8 ms
- Average upload time: 46.68 ms
- Performance requirement: PASS

## Uploaded Documents

| File | Chunks | Vectors | Upload ms |
| --- | ---: | ---: | ---: |
| leave_policy.txt | 1 | 1 | 16.85 |
| expense_policy.docx | 1 | 1 | 46.48 |
| product_manual.pdf | 1 | 1 | 227.52 |
| vpn_faq.txt | 1 | 1 | 10.12 |
| security_policy.docx | 1 | 1 | 21.25 |
| support_faq.pdf | 1 | 1 | 12.93 |
| onboarding_guide.txt | 1 | 1 | 12.68 |
| sla_policy.docx | 1 | 1 | 25.63 |

## Question Results

| # | Scenario | Expected source | Retrieval | Answer | Response ms |
| ---: | --- | --- | --- | --- | ---: |
| 1 | HR policy | leave_policy.txt | PASS | PASS | 7.8 |
| 2 | HR policy | leave_policy.txt | PASS | PASS | 7.28 |
| 3 | HR policy | leave_policy.txt | PASS | FAIL | 7.04 |
| 4 | Finance policy | expense_policy.docx | PASS | PASS | 6.76 |
| 5 | Finance policy | expense_policy.docx | PASS | PASS | 6.34 |
| 6 | Finance policy | expense_policy.docx | PASS | PASS | 6.89 |
| 7 | Product manual | product_manual.pdf | PASS | PASS | 5.97 |
| 8 | Product manual | product_manual.pdf | PASS | PASS | 7.63 |
| 9 | Product manual | product_manual.pdf | PASS | PASS | 6.9 |
| 10 | IT FAQ | vpn_faq.txt | PASS | PASS | 7.04 |
| 11 | IT FAQ | vpn_faq.txt | PASS | PASS | 6.67 |
| 12 | IT FAQ | vpn_faq.txt | PASS | PASS | 6.88 |
| 13 | Security policy | security_policy.docx | FAIL | FAIL | 5.58 |
| 14 | Security policy | security_policy.docx | PASS | PASS | 7.66 |
| 15 | Security policy | security_policy.docx | PASS | PASS | 6.9 |
| 16 | Support FAQ | support_faq.pdf | PASS | PASS | 6.21 |
| 17 | Support FAQ | support_faq.pdf | FAIL | FAIL | 6.96 |
| 18 | Onboarding | onboarding_guide.txt | PASS | PASS | 6.3 |
| 19 | SLA | sla_policy.docx | PASS | PASS | 6.41 |
| 20 | SLA | sla_policy.docx | PASS | PASS | 5.89 |

## Issues And Solutions

- External LLM calls were replaced with a deterministic local model for repeatable integration testing.
- The test corpus is ASCII English to avoid Windows console encoding problems during automated verification.
- Chroma telemetry warnings may appear in the console, but they do not affect test results.

## Conclusion

The core technical choices are feasible when retrieval and answer accuracy meet the thresholds in the automated test.
