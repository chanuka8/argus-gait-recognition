# Phase 16 — Thesis Writing Guide and Report Index

## 17.1 Suggested Thesis Chapter Structure

| Chapter | Content | Primary Source Documents |
|---|---|---|
| 1. Introduction | Background, motivation, research questions, objectives | `11_objective_rq_mapping.md` |
| 2. Literature Review | Gait recognition, GEI, biometric security, CASIA-B | `16_algorithm_reference.md` |
| 3. System Design & Architecture | Architecture diagrams, module design, design decisions | `02_system_architecture.md`, `03_gait_pipeline.md` |
| 4. Implementation | Pipeline stages, model architecture, matching logic | `03_gait_pipeline.md`, `04_model_architecture.md`, `06_gallery_and_matching.md` |
| 5. Security Architecture | Threat model, security controls, audit logging | `08_security_and_privacy.md` |
| 6. Multi-Camera System | Multi-camera design, streaming, intelligence layer | `07_multi_camera_architecture.md` |
| 7. Evaluation Methodology | Subject-disjoint protocol, leakage prevention, metrics | `05_dataset_and_preprocessing.md`, `10_performance_metrics.md` |
| 8. Results & Discussion | Performance results, cross-view analysis, limitations | `10_performance_metrics.md`, `15_cross_view_and_openset.md` |
| 9. Conclusion & Future Work | Contributions, limitations, research gaps | `12_research_contributions.md`, `13_limitations_and_gaps.md` |

## 17.2 Key Tables for Thesis

| Table | Content | Source |
|---|---|---|
| Pipeline Stage Summary | 13-stage pipeline with algorithms and outputs | `03_gait_pipeline.md` §3.2 |
| Model Layer Architecture | Layer-by-layer CNN specification | `04_model_architecture.md` §4.2 |
| Training Configuration | Hyperparameters and loss functions | `04_model_architecture.md` §4.5 |
| Subject-Disjoint Split | Train/Val/Test partition | `05_dataset_and_preprocessing.md` §5.2 |
| Decision Thresholds | Adaptive policy parameters | `06_gallery_and_matching.md` §6.5 |
| Closed-Set Results | Rank-1/5/10, condition-wise | `10_performance_metrics.md` §10.1 |
| Open-Set Results | ROC-AUC, EER, FAR/FRR | `10_performance_metrics.md` §10.2 |
| Cross-View Matrix | 11×11 accuracy matrix | `15_cross_view_and_openset.md` §15.1 |
| Security Control Matrix | Controls and gaps | `08_security_and_privacy.md` §8.1 |
| STRIDE Threat Model | Threat analysis | `08_security_and_privacy.md` §8.2 |
| Contribution Summary | 7 contributions with evidence | `12_research_contributions.md` §12.1 |
| Limitation Catalogue | 18 confirmed limitations | `13_limitations_and_gaps.md` §13.1 |

## 17.3 Key Figures for Thesis

| Figure | Description | Data Source | Reproducible |
|---|---|---|---|
| System Architecture Diagram | End-to-end pipeline flow | `02_system_architecture.md` §2.1 | Mermaid diagram |
| Data Flow Diagram | Frame → embedding → decision | `02_system_architecture.md` §2.5 | Mermaid diagram |
| Multi-Camera Architecture | Shared/isolated resources | `02_system_architecture.md` §2.3 | Mermaid diagram |
| Model Architecture | CNN layer specification | `04_model_architecture.md` §4.2 | Mermaid/manual |
| Training Loss/Accuracy Curves | 50-epoch training history | `runs/exp_001/metrics.json` | Script |
| Cross-View Heatmap | 11×11 accuracy matrix as heatmap | `cross_view_matrix.csv` | Script |
| CMC Curve | Rank-1 to Rank-20 | `closed_set_eval_report.json` | Script |
| ROC Curve | TAR vs FAR | `open_set_scores.json` | Script |
| Condition Bar Chart | NM/BG/CL accuracy comparison | `closed_set_eval_report.json` | Script |
| Adaptive Decision Flow | Decision policy flowchart | `06_gallery_and_matching.md` §6.5 | Mermaid diagram |
| Trust Boundary Diagram | Security architecture | `08_security_and_privacy.md` §8.2 | Mermaid diagram |

## 17.4 Report File Index

| # | File | Phase | Content |
|---|---|---|---|
| 01 | `01_repository_audit.md` | Phase 1 | Repository structure, environment, dependencies, component classification |
| 02 | `02_system_architecture.md` | Phase 2 | Architecture diagrams, module map, technology stack |
| 03 | `03_gait_pipeline.md` | Phase 2 | 13-stage pipeline analysis with algorithms and parameters |
| 04 | `04_model_architecture.md` | Phase 3 | ByGaitLight CNN, training config, parameter count, ArcFace |
| 05 | `05_dataset_and_preprocessing.md` | Phase 4 | CASIA-B dataset, subject split, leakage analysis |
| 06 | `06_gallery_and_matching.md` | Phase 5 | Gallery storage, cosine matching, centroid matching, decision policy |
| 07 | `07_multi_camera_architecture.md` | Phase 6 | Multi-camera pipeline, streaming, intelligence layer |
| 08 | `08_security_and_privacy.md` | Phase 7 | STRIDE threat model, security controls, privacy analysis |
| 09 | `09_testing_and_code_quality.md` | Phase 8 | Test inventory, CI/CD, code quality, placeholders |
| 10 | `10_performance_metrics.md` | Phase 9 | All evaluation results with validity classification |
| 11 | `11_objective_rq_mapping.md` | Phase 10 | Objective-evidence matrix, research question mapping |
| 12 | `12_research_contributions.md` | Phase 11 | 7 identified contributions with evidence |
| 13 | `13_limitations_and_gaps.md` | Phase 12 | 18 limitations, 12 research gaps, threats to validity |
| 14 | `14_reproducibility.md` | Phase 13 | Evidence chains, configuration traceability |
| 15 | `15_cross_view_and_openset.md` | Phase 14 | Detailed cross-view matrix, condition analysis, CMC, open-set |
| 16 | `16_algorithm_reference.md` | Phase 15 | Mathematical formulas and algorithm definitions |
| 17 | `17_thesis_writing_guide.md` | Phase 16 | Chapter mapping, table/figure index, report index |
| 18 | `FINAL_THESIS_TECHNICAL_REPORT.md` | Phase 17 | Master summary document |

## 17.5 Writing Recommendations

### Do's
- ✅ Cite every metric with its source file path
- ✅ Clearly label preliminary results as such
- ✅ Include the subject-disjoint protocol description even if clean results are pending
- ✅ Discuss both implemented and planned security features, clearly distinguishing them
- ✅ Include the leakage analysis as evidence of methodological awareness
- ✅ Use the cross-view matrix as evidence of view-angle challenges
- ✅ Include system architecture diagrams with module-level detail

### Don'ts
- ❌ Do not report the 86.89% Rank-1 as a clean subject-disjoint result
- ❌ Do not claim Zero-Trust compliance
- ❌ Do not claim template encryption or protection
- ❌ Do not claim real-world CCTV deployment validation
- ❌ Do not omit the training leakage disclosure
- ❌ Do not report validation accuracy (80.14%) as test performance
- ❌ Do not compare directly with published CASIA-B results without matching protocols
