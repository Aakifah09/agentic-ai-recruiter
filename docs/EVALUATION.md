# Evaluation

```
python scripts/run_evaluation.py                                                # rules baseline (~5 s)
python scripts/run_evaluation.py --provider ollama:llama3.2:latest --stages jd --no-fairness --name llama-jd
python scripts/run_evaluation.py --listed-credit 0.9 --name lenient-listing      # scoring-policy ablation
```

Every run is stored in the `experiments` table (and as JSON in `data/eval/results/`) and appears on the
**Evaluation** page, so changes to prompts, models or scoring policy can be compared side by side.
`tests/test_workflow.py::test_evaluation_quality_gate` turns the baseline into a regression test.

## Dataset

`data/samples/` holds 3 job descriptions (AI engineer, data analyst, QA automation engineer) and 11 fictional resumes.
The resumes cover strong, partial and unrelated profiles, plus two adversarial cases:

- **r11 – keyword stuffer:** a skills list copied from the AI job ad, with experience in customer service only.
- **Data-analyst JD with planted bias:** "young, energetic rockstar", "native English speaker", "culture fit".

`data/eval/gold.json` contains, for every job, the essential and desirable skills and a graded relevance (0–3) for all
11 resumes, and for every resume its skill set.

## Metrics

| Group | Metric | Meaning |
|---|---|---|
| Extraction | JD skill P/R/F1, importance accuracy | Did Agent 1 find the requirements and classify essential vs desirable correctly? |
| | Resume skill P/R/F1 | Did Agent 2 find the skills? (micro-averaged per resume) |
| Ranking | NDCG@3, NDCG@5, Precision@3, Spearman ρ, pairwise accuracy | Does the shortlist order match graded human judgement? |
| Responsible AI | Evidence grounding rate | Share of evidence excerpts found verbatim in the source |
| | PII leak rate | Share of candidate emails present anywhere in the output |
| | Bias-flag recall | Planted exclusionary JD phrases detected |
| | Counterfactual max / mean score shift | Score change when only name, gender, nationality and DOB change |
| Efficiency | Latency per candidate | Wall-clock time for the full workflow |

## Results (Windows 11, CPU only)

| Metric | Rules baseline | llama3.2 3B, JD agent hybrid¹ | Rules, listed-credit 0.9 |
|---|---|---|---|
| JD skill F1 | **0.977** | 0.949 | 0.977 |
| JD importance accuracy | 1.000 | 1.000 | 1.000 |
| Resume skill F1 | 0.977 | – | 0.977 |
| NDCG@3 | 0.991 | – | 0.991 |
| Precision@3 | 0.889 | – | 0.889 |
| Spearman ρ | 0.891 | – | **0.917** |
| Pairwise accuracy | 0.975 | – | **0.991** |
| Evidence grounding | 1.000 | 1.000 | 1.000 |
| PII leak rate | 0.000 | 0.000 | 0.000 |
| Bias-flag recall | 1.000 | 1.000 | 1.000 |
| Counterfactual max shift | 0.0 | not run | 0.0 |
| Latency / candidate | 0.06–0.17 s | 8.6 s | 0.06–0.17 s |

¹ The LLM run predates the implied-skills change (FastAPI → REST APIs) to the resume agent, so only its JD
metrics are comparable. They come from the JD agent alone, which that change does not touch.

**Improvement log:** an end-to-end Ollama run produced the interview question *"How does the lack of REST APIs in
your previous work affect…"* for a candidate who had built FastAPI microservices. That exposed a taxonomy gap. Adding
implied skills (FastAPI/Flask/Django → REST APIs, pytest/Selenium → Software Testing) raised pairwise accuracy from
0.948 to 0.975 and Spearman ρ from 0.850 to 0.891.

### Findings

1. **The small local LLM did not beat the rules on extraction for this data.** llama3.2 added a few plausible but
   ungraded skills, which lowered precision (0.93 vs 0.96). Before reconciliation it also labelled every
   "nice to have" skill as essential. The hybrid design (document structure decides importance) brought importance
   accuracy back to 1.0. The LLM adds most value on prose-heavy job ads and resumes that the taxonomy cannot cover,
   which this sample set barely contains.
2. **Evidence weighting is the key design decision, and it is a trade-off.** Raising the credit for listed-only skills
   from 0.5 to 0.9 improves every ranking metric here, because several genuine candidates list skills that are
   evidenced elsewhere in their resume. It also moves the keyword-stuffing resume from **9th to 4th** for the AI role
   (score 37 → 60). The default stays at 0.5 because, in screening, rewarding unevidenced claims is the worse
   failure. Recruiters can change it per run in *Scoring policy*.
3. **Remaining ranking errors are explainable.** For the QA role, the senior AI engineer (gold grade 1, score 89)
   ranks above the Java SDET (grade 2, score 82). The AI engineer has demonstrated pytest, CI/CD and Python, while the
   SDET's Java + Selenium stack misses the Python/pytest essentials. A human might weigh the dedicated testing career more
   heavily, which is exactly what the review step is for.

### Threats to validity

- **Small and self-labelled.** 33 job–resume pairs, labelled by the same developer who wrote the skill taxonomy and
  the resumes. The extraction scores in particular are optimistic; expect noticeably lower recall on real resumes
  with unusual phrasing, tables or non-tech domains.
- **Synthetic, clean text.** Real PDFs bring multi-column layouts, headers and footers, and scanned images (which this
  tool rejects with an OCR message).
- **LLM runs are slow on CPU,** so only the JD agent was evaluated end to end with an LLM. Resume and interview agents
  were tested functionally (see the README), not scored.

Recommended next steps: label 50–100 anonymised real job–resume pairs with two independent raters, report
inter-rater agreement, and evaluate the LLM resume agent on prose-heavy resumes, where rules are weakest.
