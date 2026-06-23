# Day 12 System Integration Test Report

## Scope

This report validates the complete offline API flow: upload documents, parse, split, store vectors, ask questions, return answers, and cite sources.

## Summary

- Uploaded documents: 8
- Evaluation questions: 20
- Retrieval accuracy: 90.00%
- Answer accuracy: 85.00%
- Average QA response time: 12.13 ms
- P95 QA response time: 15.21 ms
- Max QA response time: 16.65 ms
- Average upload time: 90.3 ms
- Performance requirement: PASS

## Uploaded Documents

| File | Chunks | Vectors | Upload ms |
| --- | ---: | ---: | ---: |
| leave_policy.txt | 1 | 1 | 33.11 |
| expense_policy.docx | 1 | 1 | 66.64 |
| product_manual.pdf | 1 | 1 | 498.11 |
| vpn_faq.txt | 1 | 1 | 18.26 |
| security_policy.docx | 1 | 1 | 30.56 |
| support_faq.pdf | 1 | 1 | 19.91 |
| onboarding_guide.txt | 1 | 1 | 18.36 |
| sla_policy.docx | 1 | 1 | 37.45 |

## Question Results

| # | Scenario | Expected source | Retrieval | Answer | Response ms |
| ---: | --- | --- | --- | --- | ---: |
| 1 | HR policy | leave_policy.txt | PASS | PASS | 16.65 |
| 2 | HR policy | leave_policy.txt | PASS | PASS | 11.16 |
| 3 | HR policy | leave_policy.txt | PASS | FAIL | 10.84 |
| 4 | Finance policy | expense_policy.docx | PASS | PASS | 12.38 |
| 5 | Finance policy | expense_policy.docx | PASS | PASS | 13.0 |
| 6 | Finance policy | expense_policy.docx | PASS | PASS | 13.99 |
| 7 | Product manual | product_manual.pdf | PASS | PASS | 15.21 |
| 8 | Product manual | product_manual.pdf | PASS | PASS | 14.98 |
| 9 | Product manual | product_manual.pdf | PASS | PASS | 11.0 |
| 10 | IT FAQ | vpn_faq.txt | PASS | PASS | 10.65 |
| 11 | IT FAQ | vpn_faq.txt | PASS | PASS | 9.72 |
| 12 | IT FAQ | vpn_faq.txt | PASS | PASS | 11.79 |
| 13 | Security policy | security_policy.docx | FAIL | FAIL | 11.73 |
| 14 | Security policy | security_policy.docx | PASS | PASS | 11.36 |
| 15 | Security policy | security_policy.docx | PASS | PASS | 11.85 |
| 16 | Support FAQ | support_faq.pdf | PASS | PASS | 12.07 |
| 17 | Support FAQ | support_faq.pdf | FAIL | FAIL | 13.32 |
| 18 | Onboarding | onboarding_guide.txt | PASS | PASS | 9.05 |
| 19 | SLA | sla_policy.docx | PASS | PASS | 10.84 |
| 20 | SLA | sla_policy.docx | PASS | PASS | 11.08 |

## Issues And Solutions

- External LLM calls were replaced with a deterministic local model for repeatable integration testing.
- The test corpus is ASCII English to avoid Windows console encoding problems during automated verification.
- Chroma telemetry warnings may appear in the console, but they do not affect test results.

## Conclusion

The core technical choices are feasible when retrieval and answer accuracy meet the thresholds in the automated test.
