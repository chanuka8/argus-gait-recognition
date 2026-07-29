# ARGUS AI Project Rules & Guidelines

## Zero False Positive Evidence-Based Reporting Policy

All repository audits, health checks, verification reports, and technical assessments must follow a strict evidence-based reporting policy:

### 1. Zero False Positive Policy
- Never assume a problem exists.
- Never infer a problem from best practices.
- Never invent improvement opportunities.
- Never generate generic recommendations.
- Every finding MUST be supported by one or more of:
  - Executed command output
  - Source code inspection
  - Test failure
  - Linter failure
  - Compile failure
  - Runtime exception
  - Benchmark evidence
  - Configuration mismatch
  - Repository artifact
  - Documented missing implementation
- If evidence does not exist, explicitly state:  
  `"No verified issues were identified."`

### 2. Classification Rules
Every finding must be classified as exactly one of:
- **Confirmed Issue**
- **Confirmed Improvement Opportunity**
- **Observation**
- **Informational**
- **Verified Healthy**
- **Unable to Verify**

Never classify speculation as Confirmed.

### 3. Reporting Criteria
Only report a problem if ALL of the following are true:
1. Evidence exists.
2. The evidence was produced during the audit.
3. The issue is reproducible.
4. The issue affects correctness, security, stability, functionality, or maintainability.

Do NOT report missing future features, optional optimizations, hypothetical scalability, or roadmap ideas as issues. Place them only in an optional section titled `Potential Future Enhancements (Not Issues)`.

### 4. Clean Health Report Structure
If zero verified issues exist and all checks pass (compile, lint, tests, benchmarks, docs, git status), the final verdict must explicitly state:

`PROJECT HEALTHY`

Do not create `Problems`, `Issues`, `Risks`, `Technical Debt`, or `Recommendations` sections unless supported by direct evidence.
