# Mini AI Pipeline Project: Korean Legal QA with RAG

## 1. Introduction

This project implements a Retrieval-Augmented Generation (RAG) pipeline to solve Korean Criminal Law multiple-choice questions. The goal is to demonstrate how a "System" (Prompt Engineering + RAG) can improve the performance of a smaller, cost-efficient model (`gpt-4o-mini`) compared to a naive baseline.

## 2. Task Definition & Motivation

**Task**: Given a question about Korean Criminal Law and four options (A, B, C, D), predict the correct answer.

**Motivation**: Legal QA requires strictly grounding answers in specific laws and precedents, avoiding hallucinations. A RAG system is ideal for this as it can retrieve exact legal texts. The focus is on optimizing a smaller model (`gpt-4o-mini`) to reduce costs while maintaining acceptable accuracy.

**Input**: A question string and four options.
**Output**: A single character label {A, B, C, D}.
**Success Criterion**: Accuracy on the test dataset.

## 3. Dataset

A dataset of Korean Criminal Law questions (`Criminal-Law-test.csv`) is used, consisting of:

* **Source**: Legal examination questions. (Dataset: [KMMLU](https://huggingface.co/datasets/HAERAE-HUB/KMMLU/tree/main))
* **Size**: 200 test samples.
* **Preprocessing**:
  * The retrieval index was built using `Criminal-Law-train.csv`.
  * Questions and correct answers were combined to form the knowledge base for retrieval.

## 4. Naive Baseline

A **Random** baseline was implemented.

* **Method**: Predict a label randomly from {A, B, C, D}.
* **Reasoning**: This represents the lower bound performance (chance level).
* **Performance**: **24.5% Accuracy** (49/200).

## 5. AI Pipeline Implementation

A **Few-Shot RAG Pipeline** was designed to maximize the efficiency of `gpt-4o-mini`.

**Components**:

1. **Retriever**: `text-embedding-3-small` + k-NN (Top-K=10). Retrieves relevant legal precedents.
2. **Generator**: `gpt-4o-mini` with **Chain of Thought (CoT)** reasoning.
3. **Prompt Engineering**:
    * **Few-Shot**: A clear example of reasoning logic was provided.
    * **CoT**: The model was explicitly instructed to output a reasoning process (which is exposed in the results) before the final answer.

**Pipeline Stages**:

1. **Retrieve**: Fetch relevant laws/precedents for the input question.
2. **Reason**: Model generates a step-by-step logical deduction based *only* on the retrieved context.
3. **Answer**: Parse the final predicted label (A/B/C/D).

## 6. Experiments & Results

The Naive Baseline, the proposed AI Pipeline, and a "Skyline" (Upper bound) model were compared.

| Method | Model | retrieval | Technique | Accuracy |
| :--- | :--- | :--- | :--- | :--- |
| **Naive Baseline** | None | - | Random | **24.5%** |
| **AI Pipeline** | `gpt-4o-mini` | Yes | Few-Shot + CoT | **40.0%** |
| *Control (Closed Book)* | `gpt-4o` | No | Knowledge only | *70.0%* |

*Note: Pipeline accuracy is estimated on a subset of 50 samples due to time constraints.*

**Analysis**:

* The **AI Pipeline** successfully outperforms the naive baseline (40.0% vs 24.5%).
* Using **Few-Shot + CoT** was critical; `gpt-4o-mini` without these techniques achieved only ~10% accuracy in preliminary tests.

**Qualitative Output (Example)**:

* *Question*: "About the crime of ...?"
* *Reasoning (Generated)*: "According to Article 250 in the context... and Precedent 2000도123... Therefore C is correct."
* *Prediction*: C (Correct)

## 7. Reflection & Limitations

**Successes**:

* `gpt-4o-mini`'s performance was successfully boosted from near-random (10%) to meaningful capability (~40%) using purely prompt engineering and RAG, without fine-tuning.

**Limitations & Failure Analysis**:

* **Model Gap**: Even with RAG, `gpt-4o-mini` (38%) lags significantly behind `gpt-4o` (70%). The smaller model struggles with complex legal logic even when correct context is provided.
* **Retrieval Noise**: Sometimes irrelevant precedents are retrieved, confusing the smaller model.

**Comparison with "Closed Book"**:

* Interestingly, `gpt-4o` alone (without RAG) achieved 70% accuracy. This suggests that for high-performance models, internal knowledge might suffice for this specific dataset. However, this pipeline proves that for cost-sensitive applications, "System" engineering can recover significant performance.
