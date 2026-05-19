# Benchmark RAG vs. Long-Context Prompting with Quantized Mistral-7B

## Project Overview

This project benchmarks Retrieval-Augmented Generation (RAG) against Long-Context Prompting using the quantized Mistral-7B-Instruct-v0.2 model.

The objective of this project is to compare:
- Answer quality
- Retrieval effectiveness
- Position sensitivity
- Latency and throughput
- Memory efficiency

The system evaluates how RAG improves performance compared to direct long-context prompting on long narrative documents.

---

## Technologies Used

- Python
- PyTorch
- Transformers
- bitsandbytes
- FAISS
- Sentence Transformers
- LangChain
- Ragas
- ROUGE
- BERTScore
- Docker

---

## Model Used

- mistralai/Mistral-7B-Instruct-v0.2
- 4-bit quantization using bitsandbytes

---

## Dataset

- DeepMind NarrativeQA Dataset

Dataset Link:
https://huggingface.co/datasets/deepmind/narrativeqa

---

## Project Structure

```text
.
├── notebooks/
├── src/
│   ├── rag_pipeline.py
│   ├── long_context_pipeline.py
│   ├── evaluation.py
│   ├── position_analysis.py
│   └── performance_benchmark.py
│
├── results/
│   ├── rag_answers.json
│   ├── long_context_answers.json
│   ├── evaluation_scores.json
│   ├── position_sensitivity_analysis.json
│   ├── position_sensitivity_chart.png
│   └── performance_metrics.json
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── README.md
├── .env.example
└── .gitignore
```

---

## RAG Pipeline

The Retrieval-Augmented Generation (RAG) pipeline performs the following steps:

1. Document Chunking
2. Embedding Generation
3. FAISS Vector Indexing
4. Semantic Retrieval
5. Context-Aware Answer Generation

The retrieved chunks are injected into the prompt before passing to the language model.

---

## Long-Context Pipeline

The long-context prompting pipeline directly feeds long document sections into the model without retrieval.

Documents are truncated to fit model context limitations.

This approach is benchmarked against RAG to compare:
- Retrieval effectiveness
- Generation quality
- Context handling

---

## Evaluation Metrics

The project evaluates model performance using:

- ROUGE-1
- ROUGE-L
- BERTScore

The evaluation compares:
- RAG pipeline outputs
- Long-context prompting outputs

---

## Position Sensitivity Analysis

The project analyzes the "Lost in the Middle" phenomenon by evaluating model performance when relevant information appears at:
- Beginning of document
- Middle of document
- End of document

A visualization chart is generated to compare performance across positions.

---

## Performance Benchmarking

The following metrics are measured:
- Latency
- Throughput
- GPU memory usage
- CPU memory usage

This helps evaluate the efficiency of quantized inference.

---

## Generated Results

The `results/` directory contains:

- rag_answers.json
- long_context_answers.json
- evaluation_scores.json
- position_sensitivity_analysis.json
- position_sensitivity_chart.png
- performance_metrics.json

---

## How to Run

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Docker Container

```bash
docker-compose up --build
```

---

## Key Findings

- RAG outperformed long-context prompting in ROUGE evaluation.
- Long-context prompting suffered from context dilution.
- The model demonstrated position sensitivity consistent with the "Lost in the Middle" research findings.
- Quantization reduced GPU memory usage while maintaining reasonable answer quality.

---

## Deployment Recommendation

RAG is recommended for production deployment because:
- Better retrieval grounding
- Lower context overhead
- Improved answer relevance
- Better scalability

Long-context prompting may be useful when:
- Retrieval infrastructure is unavailable
- Entire document reasoning is required

---
