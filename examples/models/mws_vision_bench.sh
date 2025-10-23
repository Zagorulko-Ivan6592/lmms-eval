#!/bin/bash

# MWS-Vision-Bench evaluation example
# This script demonstrates how to evaluate models on the MWS-Vision-Bench dataset

echo "Running MWS-Vision-Bench evaluation..."

# Example 1: OpenAI GPT-4o-mini
echo "Evaluating with GPT-4o-mini..."
python3 -m lmms_eval \
    --model openai_compatible \
    --model_args "api_key=$OPENAI_API_KEY,api_base=https://api.openai.com/v1,model_name=gpt-4o-mini" \
    --tasks mws_vision_bench_validation \
    --batch_size 10 \
    --log_samples \
    --output_path ./results/mws_vision_bench_gpt4o_mini

# Example 2: Qwen2.5-VL-7B (if using vLLM)
echo "Evaluating with Qwen2.5-VL-7B..."
python3 -m lmms_eval \
    --model openai_compatible \
    --model_args "api_key=dummy,api_base=http://localhost:8000/v1,model_name=Qwen/Qwen2.5-VL-7B-Instruct" \
    --tasks mws_vision_bench_validation \
    --batch_size 5 \
    --log_samples \
    --output_path ./results/mws_vision_bench_qwen25vl_7b

# Example 3: LLaVA-1.5 (if available)
echo "Evaluating with LLaVA-1.5..."
python3 -m lmms_eval \
    --model llava \
    --model_args "pretrained=liuhaotian/llava-v1.5-7b" \
    --tasks mws_vision_bench_validation \
    --batch_size 4 \
    --log_samples \
    --output_path ./results/mws_vision_bench_llava_15

echo "MWS-Vision-Bench evaluation completed!"
echo "Results saved in ./results/mws_vision_bench_*"
