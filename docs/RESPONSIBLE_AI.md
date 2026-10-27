# Responsible AI notes

Recruitment is a high-stakes domain. The EU AI Act lists AI used for recruitment and candidate evaluation as
**high-risk**, and New York City Local Law 144 requires bias audits for automated employment decision tools. This
project is a portfolio demonstration on fictional data. The design choices below show how such a tool should be
built; they are not a compliance certification.

## What the system will not do

- **Reject or rank anyone out.** The output schema only has three advisory labels. Every candidate stays in the list
  and every progression decision is a recruiter action, stored with reviewer, notes and timestamp in an append-only
  audit table.
- **See protected characteristics.** `privacy.redact()` runs before any agent or LLM sees a resume. It removes the
  name, email, phone, URLs and address, plus any line stating date of birth, age, gender, nationality, citizenship,
  religion, ethnicity, marital status, children, disability or visa status. It also neutralises gendered pronouns.
- **Invent evidence.** Every excerpt shown to a recruiter must be found (near-)verbatim in the redacted resume
  (`textutils.is_grounded`). LLM skill claims without a verifiable quote are dropped and counted in the parse warnings.
- **Let an LLM set the score.** Scores come from a transparent evidence table with weights the recruiter can see and
  edit.

## What the system checks

| Check | Where | Action |
|---|---|---|
| Exclusionary JD wording ("rockstar", "young", "native speaker", "culture fit", gendered pronouns, age limits) | JD Analyst + Guardrail | Flagged as a warning with a suggested alternative; never auto-edited |
| Interview questions about protected topics | Guardrail | Question removed; issue logged as *blocked* |
| Narrative text mentioning protected topics or decision words ("reject", "unsuitable") | Guardrail | Sentence removed; issue logged |
| LLM narrative contradicting the evidence table | Evidence Matcher | Narrative discarded; rule-based summary used |
| Contact details leaking into evidence | Guardrail | Re-redacted |

## Measured in the evaluation suite

- **Counterfactual invariance:** each resume is re-scored with three different names, genders, nationalities and an
  added date of birth. The maximum score change is **0.0** on the sample set.
- **PII leak rate:** share of candidate emails that appear anywhere in the serialised report: **0.0**.
- **Evidence grounding rate: 1.0.**
- **Bias-flag recall:** all 4 planted exclusionary phrases in the sample data-analyst JD are flagged.

## Known limitations

- Redaction is pattern-based. Names that are not on the first line or in a `Name:` field, and indirect signals
  (graduation years, a women's college, a religious society) can survive. A production system would add a named-entity
  recogniser and a human check of redaction quality.
- Proxy bias remains possible. Requirements such as "years of experience" correlate with age, and requirements copied
  from historic job ads can encode past bias. The tool surfaces requirements and weights so recruiters can challenge
  them; it cannot decide whether a requirement is justified.
- The skill taxonomy is English-centric and tech-weighted. Resumes from other domains or languages will be
  under-extracted, which shows up as *low confidence*, not as a low-quality candidate. Recruiters should treat low
  confidence as "read the resume yourself".
- The counterfactual test covers the rule-based agents. LLM agents receive only redacted text, but their outputs are
  not counterfactually tested in the default suite because of CPU cost.
- The ground truth was written by the developer alongside the taxonomy, so the evaluation numbers are optimistic. See
  [EVALUATION.md](EVALUATION.md).
