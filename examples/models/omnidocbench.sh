#!/bin/bash

# OmniDocBench evaluation example
# This script demonstrates how to evaluate models on the OmniDocBench dataset

echo "Running OmniDocBench evaluation..."

# Example 1: End-to-end evaluation with GPT-4o
echo "Evaluating end-to-end with GPT-4o..."
python3 -m lmms_eval \
    --model openai_compatible \
    --model_args "api_key=$OPENAI_API_KEY,api_base=https://api.openai.com/v1,model_name=gpt-4o" \
    --tasks omnidocbench_end2end \
    --batch_size 5 \
    --log_samples \
    --output_path ./results/omnidocbench_gpt4o_end2end

# Example 2: Layout detection with Qwen2.5-VL-72B
echo "Evaluating layout detection with Qwen2.5-VL-72B..."
python3 -m lmms_eval \
    --model openai_compatible \
    --model_args "api_key=dummy,api_base=http://localhost:8000/v1,model_name=Qwen/Qwen2.5-VL-72B-Instruct" \
    --tasks omnidocbench_layout_detection \
    --batch_size 3 \
    --log_samples \
    --output_path ./results/omnidocbench_qwen25vl_72b_layout

# Example 3: Table recognition with Gemini-2.5-Pro
echo "Evaluating table recognition with Gemini-2.5-Pro..."
python3 -m lmms_eval \
    --model openai_compatible \
    --model_args "api_key=$GOOGLE_API_KEY,api_base=https://generativelanguage.googleapis.com/v1beta,model_name=gemini-2.5-pro" \
    --tasks omnidocbench_table_recognition \
    --batch_size 5 \
    --log_samples \
    --output_path ./results/omnidocbench_gemini25pro_table

# Example 4: Formula recognition with Claude-4.5-Sonnet
echo "Evaluating formula recognition with Claude-4.5-Sonnet..."
python3 -m lmms_eval \
    --model openai_compatible \
    --model_args "api_key=$ANTHROPIC_API_KEY,api_base=https://api.anthropic.com/v1,model_name=claude-4.5-sonnet" \
    --tasks omnidocbench_formula_recognition \
    --batch_size 5 \
    --log_samples \
    --output_path ./results/omnidocbench_claude45_formula

# Example 5: Text OCR with InternVL3-78B
echo "Evaluating text OCR with InternVL3-78B..."
python3 -m lmms_eval \
    --model openai_compatible \
    --model_args "api_key=dummy,api_base=http://localhost:8000/v1,model_name=OpenGVLab/InternVL3-78B" \
    --tasks omnidocbench_text_ocr \
    --batch_size 2 \
    --log_samples \
    --output_path ./results/omnidocbench_internvl3_78b_text

# Example 6: All tasks with a single model
echo "Evaluating all OmniDocBench tasks with GPT-4o-mini..."
python3 -m lmms_eval \
    --model openai_compatible \
    --model_args "api_key=$OPENAI_API_KEY,api_base=https://api.openai.com/v1,model_name=gpt-4o-mini" \
    --tasks omnidocbench \
    --batch_size 3 \
    --log_samples \
    --output_path ./results/omnidocbench_gpt4o_mini_all

echo "OmniDocBench evaluation completed!"
echo "Results saved in ./results/omnidocbench_*"
