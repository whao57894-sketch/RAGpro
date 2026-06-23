# Day 12 System Integration Test Report

## Scope

This report validates the complete offline API flow: upload documents, parse, split, store vectors, ask questions, return answers, and cite sources.

## Summary

- Uploaded documents: 8
- Evaluation questions: 20
- Retrieval accuracy: 90.00%
- Answer accuracy: 85.00%
- Average QA response time: 11.92 ms
- P95 QA response time: 13.29 ms
- Max QA response time: 14.44 ms
- Average upload time: 92.39 ms
- Performance requirement: PASS

## Uploaded Documents

| File | Chunks | Vectors | Upload ms |
| --- | ---: | ---: | ---: |
| leave_policy.txt | 1 | 1 | 21.45 |
| expense_policy.docx | 1 | 1 | 64.44 |
| product_manual.pdf | 1 | 1 | 470.48 |
| vpn_faq.txt | 1 | 1 | 18.2 |
| security_policy.docx | 1 | 1 | 54.89 |
| support_faq.pdf | 1 | 1 | 31.9 |
| onboarding_guide.txt | 1 | 1 | 20.31 |
| sla_policy.docx | 1 | 1 | 57.47 |

## Question Results

| # | Scenario | Expected source | Retrieval | Answer | Response ms |
| ---: | --- | --- | --- | --- | ---: |
| 1 | HR policy | leave_policy.txt | PASS | PASS | 14.44 |
| 2 | HR policy | leave_policy.txt | PASS | PASS | 12.73 |
| 3 | HR policy | leave_policy.txt | PASS | FAIL | 13.29 |
| 4 | Finance policy | expense_policy.docx | PASS | PASS | 11.59 |
| 5 | Finance policy | expense_policy.docx | PASS | PASS | 12.19 |
| 6 | Finance policy | expense_policy.docx | PASS | PASS | 12.73 |
| 7 | Product manual | product_manual.pdf | PASS | PASS | 11.67 |
| 8 | Product manual | product_manual.pdf | PASS | PASS | 11.43 |
| 9 | Product manual | product_manual.pdf | PASS | PASS | 10.87 |
| 10 | IT FAQ | vpn_faq.txt | PASS | PASS | 12.33 |
| 11 | IT FAQ | vpn_faq.txt | PASS | PASS | 10.33 |
| 12 | IT FAQ | vpn_faq.txt | PASS | PASS | 10.3 |
| 13 | Security policy | security_policy.docx | FAIL | FAIL | 10.39 |
| 14 | Security policy | security_policy.docx | PASS | PASS | 10.86 |
| 15 | Security policy | security_policy.docx | PASS | PASS | 13.04 |
| 16 | Support FAQ | support_faq.pdf | PASS | PASS | 11.74 |
| 17 | Support FAQ | support_faq.pdf | FAIL | FAIL | 11.77 |
| 18 | Onboarding | onboarding_guide.txt | PASS | PASS | 13.0 |
| 19 | SLA | sla_policy.docx | PASS | PASS | 12.26 |
| 20 | SLA | sla_policy.docx | PASS | PASS | 11.42 |

## Issues And Solutions

- External LLM calls were replaced with a deterministic local model for repeatable integration testing.
- The test corpus is ASCII English to avoid Windows console encoding problems during automated verification.
- Chroma telemetry warnings may appear in the console, but they do not affect test results.

## Conclusion

The core technical choices are feasible when retrieval and answer accuracy meet the thresholds in the automated test.
