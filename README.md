# ModelOps – DAT CS5903 Project

## Project Overview

This project focuses on benchmarking AI models for extracting structured information from scanned documents and handwritten forms while exploring privacy-preserving AI solutions for sensitive data.

The main goal is to identify an effective model for document extraction that can support future deployment within an organization's private infrastructure.

## Models Used

- **DeepSeek-VL2:** Directly extracts information from document images.
- **Qwen2.5-VL:** Uses vision-language understanding to extract structured information in JSON format.
- **PaddleOCR + Qwen2.5-VL:** Combines OCR text recognition with vision-language extraction.

## Datasets

The project evaluates the models on five datasets:

- Form600
- FUNSD
- SROIE (Invoices/Receipts)
- IDRBT Cheque
- Nigeria Healthcare Records

## Evaluation

Model predictions are compared against ground-truth annotations using metrics such as Field-Level Accuracy, Precision, Recall, F1-score, ANLS, and Document Exact Match.

## Repository Contents

- `Datasets/` – Evaluation datasets.
- `Deepseek_VL2/` – DeepSeek-VL2 project files.
- `Deepseek_evaluation/` and `Deepseek_predictions/` – DeepSeek evaluation and outputs.
- `Qwen2.5_evaluation/` and `Qwen2.5_predictions/` – Qwen2.5-VL evaluation and outputs.
- `Pipeline_padderOCR_Qwen2.5/` – OCR-assisted pipeline implementation.
- `Pipeline_evaluation/` and `Pipeline_predictions/` – Pipeline evaluation and outputs.
- `Metrics.py` – Evaluation script.

## Project Goal

Compare the three approaches, identify their strengths and limitations, and determine a suitable model for further development toward private, domain-specific AI applications.

## Team 12

- Telugu Vinay Manikumar - CS26MTECH14007
- Sanjay Reddy Buttreddy - CS26MTECH14013
- Rahul Madhogarhia - CS24MTECH12004

**Institution:** Indian Institute of Technology Hyderabad
