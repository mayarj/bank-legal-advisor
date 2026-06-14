# Evaluation & Benchmarking Plan — Bank Legal Advisor

> This document defines how we measure whether the RAG pipeline and agents are doing their
> job correctly — not whether the code runs, but whether the system finds the right legal
> information and reasons over it accurately.

---

## 1. What We Are Measuring

The system has two distinct intelligent layers that must each be evaluated independently
before trusting their combined output.

| Layer | Core question |
|---|---|
| **RAG pipeline** | Does the retriever surface the correct legislation articles for a given query? |
| **Legal Agent** | Does the agent reason over those articles correctly and produce an accurate, well-cited answer? |
| **Loan Agent** | Does the agent apply the legal reasoning correctly to a loan application and reach the right risk decision? |

---

## 2. RAG Pipeline Evaluation

The RAG pipeline covers everything from query to retrieved articles:
query rewriting → hybrid/semantic/exact search → article fetch → metadata filtering.

### 2.1 Building a Retrieval Ground-Truth Dataset

Before running any metric, we need a labelled dataset. Each entry has:

- **Query** — a natural-language question a bank officer would ask
- **Relevant articles** — a hand-labelled list of `(legislation_code, article_number)` pairs
  that a domain expert confirms are the correct articles for that query
- **Status** — whether only `in_effect` articles are acceptable for this query

**Minimum dataset size:** 50–100 queries, spanning the legislation corpus.
Categories to cover:

| Category | Example query |
|---|---|
| Direct article lookup | "What does Article 14 of LAW-88-2003 say about collateral?" |
| Topic search | "What are the reserve requirements for commercial banks?" |
| Multi-article | "What are the conditions for loan restructuring?" |
| Relationship-aware | "Was Article 5 of LAW-12-2010 amended after 2015?" |
| Negative case | "What are the rules for cryptocurrency trading?" *(no relevant article exists)* |
| Ambiguous | "What are the penalties?" *(should trigger clarification)* |

---

### 2.2 Retrieval Metrics

Run each query through the retriever and compare the returned articles against the ground truth.

#### Precision@K
*Of the top K articles returned, what fraction are actually relevant?*

```
Precision@K = (relevant articles in top K) / K
```

We evaluate at K = 3, 5, and 10.
A well-tuned system should reach **Precision@5 ≥ 0.70** for a banking legislation corpus.

#### Recall@K
*Of all relevant articles for this query, what fraction did the retriever find in its top K?*

```
Recall@K = (relevant articles in top K) / (total relevant articles for query)
```

High recall is critical here — a missed article can mean missing a key legal constraint.
Target: **Recall@10 ≥ 0.80**.

#### Mean Reciprocal Rank (MRR)
*How high up in the list does the first correct article appear?*

```
MRR = average of (1 / rank of first correct article) across all queries
```

A score of 1.0 means the top result is always correct. Target: **MRR ≥ 0.65**.

#### Hit Rate@K
*For what percentage of queries did at least one correct article appear in the top K?*

This is the most practical metric — if the agent gets even one relevant article it can
often recover. Target: **Hit Rate@5 ≥ 0.85**.

---

### 2.3 Query Rewriting Evaluation

The retriever optionally rewrites the user query before searching. We measure its impact by
running the same 50–100 queries with and without rewriting and comparing the metrics above.

Expected outcome: rewriting should improve MRR and Recall for conversational or vague
queries. If rewriting hurts precise technical queries (e.g. exact article lookups), we
document that and tune the rewrite prompt accordingly.

---

### 2.4 Hybrid vs. Semantic vs. Exact Search Breakdown

The system uses three search modes. We report the metrics above for each mode separately:

| Mode | Strength | When it should win |
|---|---|---|
| Semantic only | Conceptual similarity | Vague or paraphrased queries |
| Exact keyword | Precise legal terms | Article numbers, law codes, specific legal phrases |
| Hybrid (semantic + BM25) | Balance | Most real-world queries |

We flag any query type where the hybrid mode is significantly outperformed by a simpler
mode — that signals a tuning opportunity in the keyword boost weight.

---

### 2.5 Metadata Filter Accuracy

The retriever can filter by `status = in_effect`. We verify:

1. **No deactivated law leaks** — queries with `status` filter must return zero articles from
   `not_in_effect` or `partially_in_effect` legislation.
2. **Recall is not over-filtered** — if a query's correct answer comes from a
   `partially_in_effect` law, we note whether the filter incorrectly excluded it.

This is a binary pass/fail check on the filter behaviour, not a metric.

---

### 2.6 Ingestion Quality Check

The ingestion step uses an LLM to extract structured data from raw PDFs. We validate the
output for a sample of legislation documents (at least 10 laws, across complexity levels).

For each document we check:

| Field | What correct looks like |
|---|---|
| `code` | Matches the official identifier on the document |
| `date` | Correct day-month-year |
| `status` | Matches current known status |
| `issuer` | Correct issuing authority |
| `subject` | Accurately summarises the law's topic (human judge: 1–5 scale) |
| `articles` | All articles present, none split across wrong boundaries |
| `relationships` | Each cited amendment/repeal is captured; no hallucinated relationships |

We report an **extraction completeness score** per document:

```
Completeness = (correctly extracted fields + articles + relationships) /
               (total expected fields + articles + relationships)
```

Target: **Completeness ≥ 0.90** per document, no law below 0.75.

---

## 3. Legal Agent Evaluation

The Legal Agent (LangGraph) goes beyond retrieval: it plans searches, traverses legislation
relationships, synthesises an answer, and self-critiques. Retrieval quality is necessary but
not sufficient — we separately evaluate what the agent does with what it retrieves.

### 3.1 Answer Correctness

For each query in the ground-truth dataset, a legal domain expert rates the agent's final
answer on two dimensions:

| Dimension | Scale | Description |
|---|---|---|
| **Factual accuracy** | 1–5 | Is the answer consistent with the cited articles and the law? |
| **Completeness** | 1–5 | Did it address all material aspects of the question? |

An answer scores a pass at ≥ 4 on both dimensions.
Target: **≥ 80% of answers pass** across the dataset.

---

### 3.2 Citation Quality

Every answer should cite the articles that support its claims. We measure:

- **Citation precision** — are all cited articles actually relevant to the answer?
  (No hallucinated article references.)
- **Citation recall** — did the answer cite all the articles it should have cited,
  based on the ground-truth labelled set?

An answer with good reasoning but wrong or missing citations is a retrieval or synthesis
failure — citation quality helps us tell these apart.

---

### 3.3 Relationship Traversal Correctness

The agent traverses parent and child legislation relationships. We create a specific test
set of queries that require following a relationship to answer correctly — for example:

> *"Is Article 7 of LAW-45-1998 still in force?"*
> (Correct answer requires finding that it was repealed by LAW-12-2010 Article 3.)

For this test set we measure:

- **Did the agent traverse the relevant relationship?** (yes/no per query)
- **Did the agent correctly evaluate the relationship illustration?** (did it skip
  irrelevant relationships and follow relevant ones?)
- **Did the final answer reflect the relationship correctly?**

Target: **≥ 75% of relationship-dependent queries answered correctly**.

---

### 3.4 Clarification Triggering

We include a set of intentionally ambiguous queries and a set of clear queries in the
evaluation. We measure:

- **False clarification rate** — percentage of clear queries where the agent unnecessarily
  asked for clarification (should be near 0%)
- **Missed clarification rate** — percentage of ambiguous queries where the agent attempted
  to answer without asking (ideally near 0%)

The target is a clarification trigger that is precise: ask when genuinely needed, proceed
confidently when the query is clear.

---

### 3.5 Critique Loop Effectiveness

The agent has a self-critique step that can send the answer back for revision. We measure:

- **Improvement rate** — for answers that failed the critique, what percentage improved
  after revision? (Measured by comparing pre- and post-critique expert ratings.)
- **Over-critique rate** — percentage of already-correct answers flagged by the critique
  step as needing revision (unnecessary loops cost latency and money).

A well-calibrated critique loop should improve weak answers and leave good answers alone.

---

## 4. Loan Agent Evaluation

The Loan Agent wraps the Legal Agent and applies its output to a concrete loan application.
Its evaluation focuses on the correctness of the final credit assessment.

### 4.1 Assessment Test Set

We construct a set of loan applications with known correct outcomes, signed off by a
credit/legal expert. Each case includes:

- **Loan details** (amount, term, collateral, applicant type)
- **Customer context** (payment history, existing loans, risk profile)
- **Expected risk level** — low / medium / high
- **Expected legal constraints** — which specific articles must be cited
- **Expected decision rationale** — the key reasoning points a correct assessment must cover

---

### 4.2 Risk Level Accuracy

The agent outputs a risk level of low, medium, or high. We compare this to the expected
level across the test set.

```
Risk Accuracy = (correctly classified cases) / (total cases)
```

We also report a **confusion matrix** — it matters whether the agent under-estimates risk
(approves a high-risk loan) more than it over-estimates (flags a safe loan). False negatives
on high-risk cases are the more damaging error.

Target: **Risk Accuracy ≥ 0.80**, with **false negative rate on high-risk cases ≤ 0.10**.

---

### 4.3 Legal Constraint Coverage

For each test case, we check whether the final assessment explicitly addresses every legal
constraint identified in the ground truth. A constraint is "covered" if the assessment
mentions the relevant article and correctly describes its implication for this loan.

```
Constraint Coverage = (legal constraints correctly addressed) /
                      (total expected constraints per case)
```

Target: **Coverage ≥ 0.85** averaged across the test set.

---

### 4.4 Customer Context Utilisation

We specifically verify that the agent uses the customer's profile, payment history, and
existing loans when they are present. For the subset of test cases with rich customer
context we check:

- Did the assessment mention the payment history?
- Did the assessment account for existing loans and total debt burden?
- Did the assessment use the credit score or risk profile signals?

This is a qualitative checklist per case, not a metric — but we flag any case where
available customer data was ignored.

---

### 4.5 Clarification Accuracy

Same principle as the Legal Agent. We create cases that genuinely need clarification
(e.g. collateral type unspecified) and cases that do not. The agent should ask exactly
when needed and never when the application is complete.

---

## 5. End-to-End Evaluation

Beyond component-level tests, we run a set of full end-to-end scenarios through the API:

| Scenario | What it tests |
|---|---|
| Simple legal query | RAG + Legal Agent full path, single article answer |
| Multi-law compliance query | Relationship traversal + synthesis |
| Ambiguous query | Clarification interrupt + resumed execution |
| Loan approval case | Loan Agent → Legal Agent → synthesis → save |
| Loan rejection case | Correct high-risk classification with legal citations |
| Non-existent regulation query | Agent gracefully reports no relevant law found |

For each scenario we record:
1. Correct final output (yes/no based on expert review)
2. Correct citations (all expected articles cited, no hallucinated ones)
3. Absence of hallucination (no invented article numbers or law codes)

---

## 6. Evaluation Cadence

| Stage | When to run | Owner |
|---|---|---|
| Retrieval metrics (Section 2) | After any change to embeddings, chunking, or vectorstore | ML engineer |
| Ingestion quality (Section 2.6) | After prompt changes to ingestion workflow | ML engineer |
| Legal Agent answer correctness (Section 3) | After any prompt or graph change | ML engineer + domain expert |
| Loan Agent risk accuracy (Section 4) | After any loan/legal agent change | ML engineer + legal reviewer |
| End-to-end scenarios (Section 5) | Before any production deployment | Whole team |

---

## 7. Failure Mode Catalogue

The following failure patterns are most likely in this system. When a benchmark run shows
degradation, match the symptom to the category first to focus the investigation.

| Symptom | Most likely cause |
|---|---|
| Low Recall — correct articles not retrieved | Embedding mismatch; query rewrite distorting intent |
| Low Precision — many irrelevant articles returned | K too high; metadata filter not applied |
| Correct articles retrieved but wrong answer | Synthesis prompt; critique not catching the error |
| Hallucinated article references in answer | Synthesis prompt allows fabrication; no citation-grounding step |
| Relationship not followed when it should be | `illustration` evaluation prompt too conservative |
| Relationship followed when it should not be | `illustration` evaluation prompt too permissive |
| Correct risk level but missing legal constraints | Legal consultation not triggered; legal questions too narrow |
| Wrong risk level | Customer context ignored; legal agent answer incorrect |
| Clarification triggered on clear queries | Ambiguity detection threshold too sensitive |
| No clarification on genuinely ambiguous queries | Ambiguity detection threshold too lenient |

---

## 8. What a "Passing" System Looks Like

A system ready for production must meet all of the following simultaneously:

- Retrieval Hit Rate@5 ≥ 0.85
- Retrieval Recall@10 ≥ 0.80
- Legal answer correctness ≥ 80% (expert-rated ≥ 4/5 on both dimensions)
- Citation precision = 1.0 (zero hallucinated article references)
- Risk level accuracy ≥ 0.80, false negative rate on high-risk ≤ 0.10
- Legal constraint coverage ≥ 0.85
- All 6 end-to-end scenarios pass

Any single failure blocks deployment and routes back to the relevant component for tuning.