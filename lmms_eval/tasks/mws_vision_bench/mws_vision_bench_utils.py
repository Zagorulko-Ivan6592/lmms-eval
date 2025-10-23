"""
MWS-Vision-Bench utility functions for lmms-eval integration.
"""

import json
import re
from typing import List, Dict, Any, Union
from lmms_eval.api.instance import Instance


def mws_vision_bench_doc_to_visual(doc: Dict[str, Any]) -> List[str]:
    """
    Extract visual information from MWS-Vision-Bench document.
    
    Args:
        doc: Document containing image path and metadata
        
    Returns:
        List of image paths
    """
    # MWS-Vision-Bench stores image path in 'image_path' field
    image_path = doc.get("image_path", "")
    if image_path:
        return [image_path]
    return []


def mws_vision_bench_doc_to_text(doc: Dict[str, Any]) -> str:
    """
    Convert MWS-Vision-Bench document to text prompt.
    
    Args:
        doc: Document containing question and metadata
        
    Returns:
        Formatted text prompt
    """
    question = doc.get("question", "")
    task_type = doc.get("type", "")
    
    # Format the prompt based on task type
    if "text grounding" in task_type.lower():
        prompt = f"Найдите указанный текст на изображении и верните его координаты в формате [x1, y1, x2, y2].\n\nВопрос: {question}"
    elif "kie" in task_type.lower() or "json" in task_type.lower():
        prompt = f"Извлеките информацию из документа и верните результат в формате JSON.\n\nВопрос: {question}"
    elif "markdown" in task_type.lower():
        prompt = f"Конвертируйте изображение документа в формат Markdown, сохраняя структуру и форматирование.\n\nВопрос: {question}"
    elif "vqa" in task_type.lower():
        prompt = f"Ответьте на вопрос о содержимом документа.\n\nВопрос: {question}"
    else:  # Default OCR task
        prompt = f"Распознайте текст на изображении.\n\nВопрос: {question}"
    
    return prompt


def mws_vision_bench_process_results(doc: Dict[str, Any], results: List[str]) -> Dict[str, Any]:
    """
    Process MWS-Vision-Bench results for evaluation.
    
    Args:
        doc: Original document
        results: Model predictions
        
    Returns:
        Processed results dictionary
    """
    if not results:
        return {"prediction": "", "ground_truth": doc.get("answers", [])}
    
    prediction = results[0] if results else ""
    ground_truth = doc.get("answers", [])
    task_type = doc.get("type", "")
    
    # Process prediction based on task type
    processed_prediction = prediction.strip()
    
    # For JSON tasks, try to extract JSON from response
    if "json" in task_type.lower() or "kie" in task_type.lower():
        try:
            # Try to find JSON in the response
            json_match = re.search(r'\{.*\}', processed_prediction, re.DOTALL)
            if json_match:
                processed_prediction = json_match.group(0)
        except:
            pass
    
    # For coordinate tasks, try to extract coordinates
    elif "grounding" in task_type.lower():
        try:
            # Try to find coordinate array in the response
            coord_match = re.search(r'\[[\d\s,\.]+\]', processed_prediction)
            if coord_match:
                processed_prediction = coord_match.group(0)
        except:
            pass
    
    return {
        "prediction": processed_prediction,
        "ground_truth": ground_truth,
        "task_type": task_type,
        "raw_prediction": prediction
    }


def mws_vision_bench_accuracy(predictions: List[str], references: List[List[Any]], **kwargs) -> Dict[str, float]:
    """
    Calculate accuracy for MWS-Vision-Bench tasks.
    
    Args:
        predictions: List of model predictions
        references: List of ground truth references
        
    Returns:
        Dictionary containing accuracy metrics
    """
    if len(predictions) != len(references):
        raise ValueError("Number of predictions and references must match")
    
    correct = 0
    total = len(predictions)
    
    # Task-specific accuracy calculation
    task_accuracies = {}
    
    for i, (pred, ref) in enumerate(zip(predictions, references)):
        # Get task type from kwargs if available
        task_type = kwargs.get('task_types', [None] * len(predictions))[i] if 'task_types' in kwargs else None
        
        is_correct = False
        
        if task_type:
            if "json" in task_type.lower() or "kie" in task_type.lower():
                # For JSON tasks, compare JSON structure
                is_correct = _compare_json_answers(pred, ref)
            elif "grounding" in task_type.lower():
                # For coordinate tasks, compare coordinate arrays
                is_correct = _compare_coordinate_answers(pred, ref)
            elif "markdown" in task_type.lower():
                # For markdown tasks, use edit distance
                is_correct = _compare_markdown_answers(pred, ref)
            else:
                # For VQA and OCR tasks, use exact match or similarity
                is_correct = _compare_text_answers(pred, ref)
        else:
            # Default comparison
            is_correct = _compare_text_answers(pred, ref)
        
        if is_correct:
            correct += 1
        
        # Track accuracy by task type
        if task_type:
            if task_type not in task_accuracies:
                task_accuracies[task_type] = {"correct": 0, "total": 0}
            task_accuracies[task_type]["total"] += 1
            if is_correct:
                task_accuracies[task_type]["correct"] += 1
    
    # Calculate overall accuracy
    overall_accuracy = correct / total if total > 0 else 0.0
    
    # Calculate task-specific accuracies
    task_specific_accuracies = {}
    for task_type, stats in task_accuracies.items():
        task_specific_accuracies[f"{task_type}_accuracy"] = stats["correct"] / stats["total"] if stats["total"] > 0 else 0.0
    
    return {
        "accuracy": overall_accuracy,
        **task_specific_accuracies
    }


def _compare_json_answers(pred: str, ref: List[Any]) -> bool:
    """Compare JSON answers with some tolerance for formatting differences."""
    try:
        pred_json = json.loads(pred) if isinstance(pred, str) else pred
        ref_json = ref[0] if ref and isinstance(ref[0], (dict, list)) else ref
        
        # Simple comparison - can be enhanced with more sophisticated JSON comparison
        return str(pred_json).strip() == str(ref_json).strip()
    except:
        return False


def _compare_coordinate_answers(pred: str, ref: List[Any]) -> bool:
    """Compare coordinate answers with some tolerance for formatting."""
    try:
        # Extract coordinates from prediction
        coord_match = re.search(r'\[[\d\s,\.]+\]', pred)
        if not coord_match:
            return False
        
        pred_coords = json.loads(coord_match.group(0))
        ref_coords = ref[0] if ref else []
        
        if len(pred_coords) != len(ref_coords):
            return False
        
        # Compare with some tolerance for floating point differences
        tolerance = 5.0
        for p, r in zip(pred_coords, ref_coords):
            if abs(float(p) - float(r)) > tolerance:
                return False
        
        return True
    except:
        return False


def _compare_markdown_answers(pred: str, ref: List[Any]) -> bool:
    """Compare markdown answers using edit distance."""
    try:
        ref_text = ref[0] if ref else ""
        
        # Simple comparison - can be enhanced with markdown-specific comparison
        pred_clean = re.sub(r'\s+', ' ', pred.strip())
        ref_clean = re.sub(r'\s+', ' ', ref_text.strip())
        
        return pred_clean == ref_clean
    except:
        return False


def _compare_text_answers(pred: str, ref: List[Any]) -> bool:
    """Compare text answers with some tolerance for formatting."""
    try:
        ref_text = ref[0] if ref else ""
        
        # Normalize text for comparison
        pred_clean = re.sub(r'\s+', ' ', pred.strip().lower())
        ref_clean = re.sub(r'\s+', ' ', ref_text.strip().lower())
        
        # Exact match or contains match
        return pred_clean == ref_clean or ref_clean in pred_clean
    except:
        return False


def mws_vision_bench_aggregate_results(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Aggregate MWS-Vision-Bench results across all samples.
    
    Args:
        results: List of result dictionaries
        
    Returns:
        Aggregated metrics
    """
    if not results:
        return {"accuracy": 0.0}
    
    # Extract task types and accuracies
    task_types = [r.get("task_type", "unknown") for r in results]
    predictions = [r.get("prediction", "") for r in results]
    ground_truths = [r.get("ground_truth", []) for r in results]
    
    # Calculate accuracy with task type information
    accuracy_metrics = mws_vision_bench_accuracy(
        predictions=predictions,
        references=ground_truths,
        task_types=task_types
    )
    
    return accuracy_metrics
