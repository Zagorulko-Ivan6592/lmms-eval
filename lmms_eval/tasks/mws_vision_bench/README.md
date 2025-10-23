# MWS-Vision-Bench Integration

This directory contains the integration of [MWS-Vision-Bench](https://github.com/mts-ai/MWS-Vision-Bench) into lmms-eval.

## Overview

MWS-Vision-Bench is the first comprehensive Russian OCR benchmark for multimodal large language models. It focuses on real-world business scenarios with authentic documents that companies actually encounter.

## Dataset Information

- **Source**: [MTSAIR/MWS-Vision-Bench](https://huggingface.co/datasets/MTSAIR/MWS-Vision-Bench) on HuggingFace
- **Language**: Russian
- **Size**: 2,580 question-answer pairs across 800 unique images
- **Split**: Validation set (public)

## Task Types

The benchmark includes 5 core task types:

1. **Text OCR** - Basic image-to-text conversion
2. **Structured OCR** - Image-to-Markdown conversion (requiring layout understanding)
3. **Text Localization** - Find and return bounding boxes for specific text
4. **Key Information Extraction** - Extract structured data (JSON format)
5. **Visual Question Answering** - Answer questions about document content

## Usage

### Basic Evaluation

```bash
python3 -m lmms_eval \
    --model openai_compatible \
    --model_args "api_key=$OPENAI_API_KEY,api_base=https://api.openai.com/v1,model_name=gpt-4o-mini" \
    --tasks mws_vision_bench_validation \
    --batch_size 10
```

### Available Tasks

- `mws_vision_bench_validation` - Full validation set evaluation

### Metrics

- `mws_vision_bench_accuracy` - Overall accuracy across all task types
- Task-specific accuracies for each of the 5 task types

## Configuration Files

- `mws_vision_bench.yaml` - Main task group configuration
- `mws_vision_bench_validation.yaml` - Validation set configuration
- `_default_template_mws_vision_bench_yaml` - Default template for all tasks
- `mws_vision_bench_utils.py` - Utility functions for data processing and evaluation

## Example Results

The benchmark provides comprehensive evaluation results including:

- Overall accuracy across all task types
- Per-task accuracy breakdown
- Support for different model types (OpenAI, vLLM, local models)

## References

- [MWS-Vision-Bench GitHub](https://github.com/mts-ai/MWS-Vision-Bench)
- [HuggingFace Dataset](https://huggingface.co/datasets/MTSAIR/MWS-Vision-Bench)
- [Habr Article (Russian)](https://habr.com/ru/companies/mts_ai/articles/953292/)
