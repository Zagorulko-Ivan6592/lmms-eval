# OmniDocBench Integration

This directory contains the integration of [OmniDocBench](https://github.com/opendatalab/OmniDocBench) into lmms-eval.

## Overview

OmniDocBench is a comprehensive benchmark for evaluating diverse document parsing in real-world scenarios. It features rich annotations for evaluation across several dimensions including end-to-end parsing, layout detection, table recognition, formula recognition, and text OCR.

## Dataset Information

- **Source**: [opendatalab/OmniDocBench](https://huggingface.co/datasets/opendatalab/OmniDocBench) on HuggingFace
- **Size**: 1,355 PDF pages covering 9 document types, 4 layout types, and 3 language types
- **Languages**: English, Simplified Chinese, Mixed
- **Document Types**: Academic papers, financial reports, newspapers, textbooks, handwritten notes, etc.

## Task Types

The benchmark supports 5 main evaluation tasks:

1. **End-to-End Evaluation** - Complete document parsing to markdown
2. **Layout Detection** - Detect and localize document elements
3. **Table Recognition** - Extract tables in HTML format
4. **Formula Recognition** - Convert formulas to LaTeX
5. **Text OCR** - Extract and recognize text content

## Usage

### End-to-End Evaluation

```bash
python3 -m lmms_eval \
    --model openai_compatible \
    --model_args "api_key=$OPENAI_API_KEY,api_base=https://api.openai.com/v1,model_name=gpt-4o" \
    --tasks omnidocbench_end2end \
    --batch_size 5
```

### Layout Detection

```bash
python3 -m lmms_eval \
    --model openai_compatible \
    --model_args "api_key=$OPENAI_API_KEY,api_base=https://api.openai.com/v1,model_name=gpt-4o" \
    --tasks omnidocbench_layout_detection \
    --batch_size 5
```

### Table Recognition

```bash
python3 -m lmms_eval \
    --model openai_compatible \
    --model_args "api_key=$OPENAI_API_KEY,api_base=https://api.openai.com/v1,model_name=gpt-4o" \
    --tasks omnidocbench_table_recognition \
    --batch_size 5
```

### Formula Recognition

```bash
python3 -m lmms_eval \
    --model openai_compatible \
    --model_args "api_key=$OPENAI_API_KEY,api_base=https://api.openai.com/v1,model_name=gpt-4o" \
    --tasks omnidocbench_formula_recognition \
    --batch_size 5
```

### Text OCR

```bash
python3 -m lmms_eval \
    --model openai_compatible \
    --model_args "api_key=$OPENAI_API_KEY,api_base=https://api.openai.com/v1,model_name=gpt-4o" \
    --tasks omnidocbench_text_ocr \
    --batch_size 5
```

### All Tasks

```bash
python3 -m lmms_eval \
    --model openai_compatible \
    --model_args "api_key=$OPENAI_API_KEY,api_base=https://api.openai.com/v1,model_name=gpt-4o" \
    --tasks omnidocbench \
    --batch_size 3
```

## Available Tasks

- `omnidocbench_end2end` - End-to-end document parsing
- `omnidocbench_layout_detection` - Layout element detection
- `omnidocbench_table_recognition` - Table structure recognition
- `omnidocbench_formula_recognition` - Mathematical formula recognition
- `omnidocbench_text_ocr` - Text content extraction
- `omnidocbench` - All tasks combined

## Metrics

### End-to-End Evaluation
- `omnidocbench_overall_score` - Overall score combining text, table, and formula metrics
- `omnidocbench_text_edit_distance` - Normalized edit distance for text
- `omnidocbench_table_teds` - TEDS (Tree Edit Distance based Similarity) for tables
- `omnidocbench_formula_cdm` - CDM (Comprehensive Distance Metric) for formulas

### Layout Detection
- `omnidocbench_layout_map` - mAP (mean Average Precision) for layout detection

### Table Recognition
- `omnidocbench_table_teds` - TEDS score for table structure

### Formula Recognition
- `omnidocbench_formula_cdm` - CDM score for formula accuracy

### Text OCR
- `omnidocbench_text_edit_distance` - Normalized edit distance for text recognition

## Configuration Files

- `omnidocbench.yaml` - Main task group configuration
- `omnidocbench_end2end.yaml` - End-to-end evaluation configuration
- `omnidocbench_layout_detection.yaml` - Layout detection configuration
- `omnidocbench_table_recognition.yaml` - Table recognition configuration
- `omnidocbench_formula_recognition.yaml` - Formula recognition configuration
- `omnidocbench_text_ocr.yaml` - Text OCR configuration
- `omnidocbench_utils.py` - Utility functions for data processing and evaluation

## Dataset Features

### Rich Annotations
- **Block-level elements**: 15 types (text paragraphs, headings, tables, figures, etc.)
- **Span-level elements**: 4 types (text lines, inline formulas, subscripts, etc.)
- **Reading order**: Annotations for document component reading order
- **Attributes**: Page-level and block-level attribute tags

### Document Types
- Academic literature
- Financial reports
- Newspapers
- Textbooks
- Handwritten notes
- Magazines
- Research reports
- Exam papers
- PPT presentations

### Layout Types
- Single column
- Double column
- Three column
- Mixed layouts

## Example Results

The benchmark provides comprehensive evaluation results including:

- Overall scores across all document types
- Per-task performance breakdown
- Attribute-level evaluation results
- Support for different model types and inference methods

## References

- [OmniDocBench GitHub](https://github.com/opendatalab/OmniDocBench)
- [HuggingFace Dataset](https://huggingface.co/datasets/opendatalab/OmniDocBench)
- [OpenDataLab Dataset](https://opendatalab.com/OpenDataLab/OmniDocBench)
- [arXiv Paper](https://arxiv.org/abs/2412.07626)
