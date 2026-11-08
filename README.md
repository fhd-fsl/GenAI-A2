# TrialMatch: Parameter-Efficient Clinical Trial Screening & Efficient Transformer Inference

> **Generative AI Assignment #2**  
> **Author:** Fahad Bin Faisal (Roll No. 23i-0614)  
> **Department of Computer Science**, FAST-NUCES, Islamabad  
> **Email:** `i230614@isb.nu.edu.pk`  
> **Hardware:** NVIDIA GeForce RTX 4050 Laptop GPU (6.0 GB VRAM)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-green.svg)](https://opensource.org/licenses/Apache-2.0)
[![Report PDF](https://img.shields.io/badge/Technical_Report-IEEE_PDF-red.svg)](report/GenAI_Assignment2_Report.pdf)
[![Video Walkthrough](https://img.shields.io/badge/Demo_Video-YouTube-red.svg)](https://youtu.be/HCHado1Rx9s)

---

## 📌 Submission Deliverables & Links

| Deliverable | Location / URL | Description |
| :--- | :--- | :--- |
| **Technical Report** | [`report/GenAI_Assignment2_Report.pdf`](report/GenAI_Assignment2_Report.pdf) | 9-page IEEE conference paper with complete theory, benchmarks, and appendices. |
| **Video Demonstration** | [YouTube Video (5--7 min)](https://youtu.be/HCHado1Rx9s) | End-to-end screencast walking through the UI, decoding lab, and benchmarks. |
| **GitHub Repository** | [github.com/fhd-fsl/GenAI-A2](https://github.com/fhd-fsl/GenAI-A2) | Complete open-source codebase with automated test suites and reproducible scripts. |
| **Adapter Weights** | [Google Drive Folder](https://drive.google.com/drive/folders/1YEafpplZJwsxZx9rfEQxkuEPuCg8xUrp?usp=sharing) | Pre-trained QLoRA weights (`adapter_model.safetensors`, 479 MB). |
| **One-Click App Launcher** | [`run_app.bat`](run_app.bat) | Windows batch file to launch the local web server and open the browser. |

---

## ⚡ Quick-Start: Running on a Fresh Clone

If someone clones this repository from scratch on a clean machine without any prior weights or local data, follow these 3 simple steps:

### 1. Clone & Install Dependencies
```bash
git clone https://github.com/fhd-fsl/GenAI-A2.git
cd GenAI-A2

# Install dependencies using pip (or 'uv sync' if using uv)
pip install -r requirements.txt
```

### 2. LoRA Adapter Weights (One-Click or Base Fallback)
The LoRA weights (`adapter_model.safetensors`, 479 MB) are hosted on Google Drive to adhere to GitHub's file-size ceiling.
* **Option A (Download Weights)**:
  ```bash
  python download_adapter_weights.py
  ```
  *(Or download directly from [Google Drive](https://drive.google.com/drive/folders/1YEafpplZJwsxZx9rfEQxkuEPuCg8xUrp?usp=sharing) and place inside `task1/outputs/adapter/`).*
* **Option B (Zero-Download Base Fallback)**:
  If the adapter weights file is not present, all scripts and the web application **automatically fall back to running the base model (`Qwen/Qwen2.5-3B-Instruct`)**. Tasks 3 (KV Cache), Task 4 (Attention), Task 5 (Speculative Decoding), and Task 6 (Web App) run without crashing!

### 3. Launch the Web Application
```bash
# Windows double-click:
run_app.bat

# Or terminal:
python task6/app.py
```
Open your browser at **`http://127.0.0.1:8000`**. The workstation will load with built-in preset clinical vignettes, live inference, and diagnostic telemetry!

---

## 🏗️ Architecture & Model Selection

* **Target Model:** `Qwen/Qwen2.5-3B-Instruct` (3.09B parameters)
  * **Quantization:** 4-bit NormalFloat (NF4) via `bitsandbytes` (~3.49 GB VRAM).
  * **Grouped-Query Attention (GQA):** 16 query heads : 2 KV heads ($8:1$ ratio), reducing KV cache to $36.0\text{ KB/token}$.
  * **Context Window:** 131,072 tokens with RoPE positional embeddings.
  * **License:** Apache 2.0 (permissive commercial and academic deployment).
* **Draft Model (Task 5):** `Qwen/Qwen2.5-0.5B-Instruct` (0.49B parameters)
  * **Precision:** Native FP16 (~0.98 GB VRAM).
  * **Vocabulary:** 151,665 tokens (**100% identical tokenizer** to Target Model, zero-overhead mapping).
* **Dual-Model Shared Footprint:** ~4.47 GB static VRAM, operating comfortably within a single 6.0 GB laptop GPU.

---

## 🔬 Tasks Overview & Verification Suite

Every task includes automated correctness verification and benchmarking suites:

### Task 1: QLoRA Fine-Tuning
- **Dataset:** `ncbi/TrialGPT-Criterion-Annotations` (1,015 criteria across 53 patients).
- **Leakage Prevention:** Fixed patient-grouped split (**37 train / 8 val / 8 test**) via `task1/outputs/train/split_manifest.json`.
- **QLoRA Config:** Rank $r=128$, $\alpha=256$, all 7 projections (`q, k, v, o, gate, up, down`), trained for 3 epochs.
- **Reproduce Data Preparation:**
  ```bash
  python data/download_dataset.py
  python data/preprocess.py
  ```

### Task 2: Verification, Evidence Retrieval & Error Taxonomy
- Evaluates 6-class eligibility classification on held-out test split ($n=206$).
- Sentence-level evidence extraction evaluated via strict 4-case null set policy.
- **Run Evaluation:**
  ```bash
  python task2/eval_task2.py --adapter-path task1/outputs/adapter
  ```
- **Results:** **73.30% Accuracy**, **0.5402 Macro-F1**, **0.8260 Evidence Macro-F1** (vs. 22.82% base model).

### Task 3: Explicit Low-Level KV-Cache Engine
- Implemented `ExplicitKVCache` in [`task3/kv_cache.py`](task3/kv_cache.py) with explicit RoPE offsets and prefill/decode separation.
- **Verify 100% Token Equivalence:**
  ```bash
  python task3/test_correctness.py
  ```
- **Run KV Benchmarks:**
  ```bash
  python task3/benchmark_kv.py
  ```
- **Results:** **$2.56\times$ speedup** on long clinical vignettes ($7.40\text{ tok/s}$ vs. $2.89\text{ tok/s}$), $108.93\text{ MB}$ saved, **100.00% exact token match**.

### Task 4: Tiled Online-Softmax Attention
- Implemented blockwise online normalizer attention in [`task4/attention.py`](task4/attention.py) without intermediate $N \times N$ matrix materialization.
- **Verify Numerical Precision:**
  ```bash
  python task4/test_attention.py
  ```
- **Run Scaling Benchmarks:**
  ```bash
  python task4/attention_benchmark.py
  ```
- **Results:** Peak memory reduced by **$95.0\%$** at $N=4096$ ($40.95\text{ MB}$ vs $824.27\text{ MB}$ SDPA), with maximum error $L_\infty \le 1.95 \times 10^{-3}$.

### Task 5: Speculative Decoding Engine
- Implemented greedy speculative drafting and parallel target verification in [`task5/speculative_decoding.py`](task5/speculative_decoding.py) with exact KV cache state rollback.
- **Verify Exact Output Equivalence (18 configurations):**
  ```bash
  python task5/test_speculative.py
  ```
- **Run Speculative Benchmarks:**
  ```bash
  python task5/benchmark_speculative.py
  ```
- **Results:** Up to **$1.75\times$ speedup** and $75.0\%$ forward passes saved on structured JSON schemas, alongside empirical documentation of the **$0.92\times$ slowdown** on high-entropy prose.

### Task 6: Unified Clinical Web Application
- Standalone single-page clinical decision support dashboard developed with `aiohttp`, vanilla JavaScript, and modern clinical styling.
- Features: Bidirectional sentence-level evidence highlights, 3 explicit aggregation rules (`POTENTIALLY_ELIGIBLE`, `UNCERTAIN_MISSING_INFO`, `INELIGIBLE`), live decoding comparisons, and printable PDF audit reports.
- **Verify Backend & API Endpoints:**
  ```bash
  python task6/test_app.py
  ```

---

## 📊 Summary of Empirical Results

| Metric / Dimension | Baseline (Unoptimized / Zero-Shot) | TrialMatch Optimized System | Empirical Gain |
| :--- | :---: | :---: | :---: |
| **Classification Accuracy** | $22.82\%$ | **$73.30\%$** | **$+50.48\text{ pp}$** |
| **Macro-Averaged F1** | $0.1365$ | **$0.5402$** | **$+0.4037$** |
| **Evidence Retrieval Macro-F1** | $0.7011$ | **$0.8260$** | **$+0.1249$** |
| **JSON Schema Adherence** | $0.0\%$ | **$100.0\%$** | **$+100.0\text{ pp}$** |
| **KV Cache Autoregressive Speed** | $2.89\text{ tok/s}$ | **$7.40\text{ tok/s}$** | **$2.56\times$ Speedup** |
| **KV Cache Token Equivalence** | Reference | $675 / 675$ | **$100.00\%$ Match** |
| **Attention Memory Footprint ($N=4096$)** | $824.27\text{ MB}$ | **$40.95\text{ MB}$** | **$95.0\%$ Reduction** |
| **Speculative Speedup (JSON Schema)** | $7.80\text{ tok/s}$ ($\gamma=0$) | **$13.69\text{ tok/s}$** ($\gamma=5$) | **$1.75\times$ Speedup** |
| **Speculative Passes Saved** | $0\%$ | **$75.0\%$** | **$-75\%$ Compute** |

---

## 📑 Recompiling the Technical Report

The report source is located in [`report/`](report/) and adheres strictly to IEEE conference formatting. To recompile the PDF:

```powershell
.\report\compile_report.bat
```
*(Executes automated passes: `pdflatex` $\to$ `bibtex` $\to$ `pdflatex` $\times 2$, producing `report/GenAI_Assignment2_Report.pdf` with 0 overfull hboxes).*

---

## ⚠️ Educational & Research Disclaimer

**TrialMatch is an educational and academic research screening prototype.** It is designed to assist clinical research coordinators in pre-screening patient records against trial eligibility protocols. It does **not** provide clinical diagnoses, medical advice, or official patient enrollment authorizations. All predictions must be reviewed by qualified medical personnel.

---

## 📜 License

- **Code:** [MIT License](LICENSE)
- **Model Weights:** [Apache 2.0 License](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct)
