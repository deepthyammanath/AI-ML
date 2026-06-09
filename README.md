# AI/ML Portfolio — Deepthy Narayanan

M.Tech student at BITS Pilani (AIMLCZG536 — Large Language Models for Generative AI).  
This repository documents hands-on projects in LLM fine-tuning, inference optimization, and production deployment.

---

## Projects

### 1. Domain LLM Adaptation & Production Optimization


End-to-end pipeline adapting a general-purpose LLM to the financial domain — from raw PDFs to a fine-tuned, quantized, production-benchmarked model.

**What I built:**
- Collected and cleaned 6 financial documents (510,000+ words) from Tesla, Apple, World Bank, and RBI
- Generated 250 instruction–response Q&A pairs from domain text
- Fine-tuned TinyLlama-1.1B using QLoRA (4-bit) — training loss dropped from 7.28 → 0.41
- Benchmarked 5 decoding strategies (Greedy, Beam Search, Top-K, Top-P, Temperature)
- Implemented speculative decoding with SmolLM2-135M as draft model
- Applied 4-bit NF4 quantization and computed production cost in ₹/1M tokens

**Key results:**

| Configuration | VRAM | Throughput | Cost/1M tokens |
|---|---|---|---|
| bfloat16 baseline | 2.61 GB | 30.65 tok/s | ₹31.72 |
| 4-bit NF4 quantized | 3.05 GB | 18.66 tok/s | ₹52.10 |
| 4-bit + Speculative Decoding | 2.79 GB | 8.44 tok/s | ₹115.19 |

**Tech stack:** Python, HuggingFace Transformers, PEFT, bitsandbytes, PyMuPDF, Kaggle (2x Tesla T4)

---

## Skills demonstrated

- Large Language Model fine-tuning (QLoRA, LoRA adapters)
- Inference optimization (quantization, speculative decoding, decoding strategies)
- Data pipeline engineering (PDF extraction, cleaning, deduplication)
- Production cost analysis for LLM serving
- GPU computing (CUDA, Tesla T4)

---

## Environment

- Platform: Kaggle (2x Tesla T4 GPUs, 30GB VRAM)
- Models: TinyLlama-1.1B, SmolLM2-135M
- Course: AIMLCZG536 — Large Language Models for Generative AI, BITS Pilani

---

## About me

M.Tech student specializing in AI/ML at BITS Pilani.  
Interested in LLM engineering, model optimization, and production AI systems.

