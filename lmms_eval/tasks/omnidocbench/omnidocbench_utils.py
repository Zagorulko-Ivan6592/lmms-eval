"""
OmniDocBench utility functions for lmms-eval integration.
"""

import json
import re
import math
from typing import List, Dict, Any, Union, Tuple
from lmms_eval.api.instance import Instance


def omnidocbench_doc_to_visual(doc: Dict[str, Any]) -> List[str]:
    """
    Extract visual information from OmniDocBench document.
    
    Args:
        doc: Document containing page info and layout detections
        
    Returns:
        List of image paths
    """
    page_info = doc.get("page_info", {})
    image_path = page_info.get("image_path", "")
    if image_path:
        return [image_path]
    return []


def omnidocbench_doc_to_text(doc: Dict[str, Any]) -> str:
    """
    Convert OmniDocBench document to text prompt.
    
    Args:
        doc: Document containing page info and layout detections
        
    Returns:
        Formatted text prompt
    """
    page_info = doc.get("page_info", {})
    layout_dets = doc.get("layout_dets", [])
    
    # Count different element types
    element_counts = {}
    for det in layout_dets:
        cat_type = det.get("category_type", "unknown")
        element_counts[cat_type] = element_counts.get(cat_type, 0) + 1
    
    # Create descriptive prompt
    prompt = "Parse this document page and convert it to structured markdown format, preserving layout, tables, formulas, and text formatting."
    
    if element_counts:
        element_desc = ", ".join([f"{count} {cat_type}" for cat_type, count in element_counts.items()])
        prompt += f"\n\nThe page contains: {element_desc}."
    
    return prompt


def omnidocbench_process_end2end_results(doc: Dict[str, Any], results: List[str]) -> Dict[str, Any]:
    """
    Process OmniDocBench end-to-end results for evaluation.
    
    Args:
        doc: Original document
        results: Model predictions
        
    Returns:
        Processed results dictionary
    """
    if not results:
        return {
            "prediction": "",
            "ground_truth": doc.get("layout_dets", []),
            "page_info": doc.get("page_info", {})
        }
    
    prediction = results[0] if results else ""
    ground_truth = doc.get("layout_dets", [])
    page_info = doc.get("page_info", {})
    
    return {
        "prediction": prediction.strip(),
        "ground_truth": ground_truth,
        "page_info": page_info,
        "raw_prediction": prediction
    }


def omnidocbench_process_layout_results(doc: Dict[str, Any], results: List[str]) -> Dict[str, Any]:
    """
    Process OmniDocBench layout detection results.
    
    Args:
        doc: Original document
        results: Model predictions
        
    Returns:
        Processed results dictionary
    """
    if not results:
        return {
            "prediction": [],
            "ground_truth": doc.get("layout_dets", []),
            "page_info": doc.get("page_info", {})
        }
    
    prediction = results[0] if results else ""
    
    # Parse prediction as JSON if possible, otherwise return as text
    try:
        pred_parsed = json.loads(prediction)
        if isinstance(pred_parsed, list):
            prediction_boxes = pred_parsed
        else:
            prediction_boxes = []
    except:
        prediction_boxes = []
    
    return {
        "prediction": prediction_boxes,
        "ground_truth": doc.get("layout_dets", []),
        "page_info": doc.get("page_info", {}),
        "raw_prediction": prediction
    }


def omnidocbench_process_table_results(doc: Dict[str, Any], results: List[str]) -> Dict[str, Any]:
    """
    Process OmniDocBench table recognition results.
    
    Args:
        doc: Original document
        results: Model predictions
        
    Returns:
        Processed results dictionary
    """
    if not results:
        return {
            "prediction": "",
            "ground_truth": "",
            "table_info": {}
        }
    
    prediction = results[0] if results else ""
    
    # Extract table information from ground truth
    layout_dets = doc.get("layout_dets", [])
    table_info = {}
    ground_truth_html = ""
    
    for det in layout_dets:
        if det.get("category_type") == "table":
            table_info = det
            ground_truth_html = det.get("html", "")
            break
    
    return {
        "prediction": prediction.strip(),
        "ground_truth": ground_truth_html,
        "table_info": table_info,
        "raw_prediction": prediction
    }


def omnidocbench_process_formula_results(doc: Dict[str, Any], results: List[str]) -> Dict[str, Any]:
    """
    Process OmniDocBench formula recognition results.
    
    Args:
        doc: Original document
        results: Model predictions
        
    Returns:
        Processed results dictionary
    """
    if not results:
        return {
            "prediction": "",
            "ground_truth": "",
            "formula_info": {}
        }
    
    prediction = results[0] if results else ""
    
    # Extract formula information from ground truth
    layout_dets = doc.get("layout_dets", [])
    formula_info = {}
    ground_truth_latex = ""
    
    for det in layout_dets:
        if det.get("category_type") == "equation_isolated":
            formula_info = det
            ground_truth_latex = det.get("latex", "")
            break
    
    return {
        "prediction": prediction.strip(),
        "ground_truth": ground_truth_latex,
        "formula_info": formula_info,
        "raw_prediction": prediction
    }


def omnidocbench_process_text_ocr_results(doc: Dict[str, Any], results: List[str]) -> Dict[str, Any]:
    """
    Process OmniDocBench text OCR results.
    
    Args:
        doc: Original document
        results: Model predictions
        
    Returns:
        Processed results dictionary
    """
    if not results:
        return {
            "prediction": "",
            "ground_truth": "",
            "text_info": {}
        }
    
    prediction = results[0] if results else ""
    
    # Extract text information from ground truth
    layout_dets = doc.get("layout_dets", [])
    text_info = {}
    ground_truth_text = ""
    
    for det in layout_dets:
        if det.get("category_type") in ["text_block", "title"] and det.get("text"):
            text_info = det
            ground_truth_text = det.get("text", "")
            break
    
    return {
        "prediction": prediction.strip(),
        "ground_truth": ground_truth_text,
        "text_info": text_info,
        "raw_prediction": prediction
    }


def omnidocbench_overall_score(predictions: List[str], references: List[List[Any]], **kwargs) -> Dict[str, float]:
    """
    Calculate overall score for OmniDocBench end-to-end evaluation.
    
    Args:
        predictions: List of model predictions (markdown)
        references: List of ground truth layout detections
        
    Returns:
        Dictionary containing overall score
    """
    if len(predictions) != len(references):
        raise ValueError("Number of predictions and references must match")
    
    text_scores = []
    table_scores = []
    formula_scores = []
    
    for pred, ref in zip(predictions, references):
        # Calculate text score (1 - normalized edit distance)
        text_score = _calculate_text_score(pred, ref)
        text_scores.append(text_score)
        
        # Calculate table score (TEDS)
        table_score = _calculate_table_score(pred, ref)
        table_scores.append(table_score)
        
        # Calculate formula score (CDM approximation)
        formula_score = _calculate_formula_score(pred, ref)
        formula_scores.append(formula_score)
    
    # Overall score = (Text + Table + Formula) / 3
    overall_score = (
        sum(text_scores) / len(text_scores) +
        sum(table_scores) / len(table_scores) +
        sum(formula_scores) / len(formula_scores)
    ) / 3
    
    return {
        "overall_score": overall_score,
        "text_score": sum(text_scores) / len(text_scores),
        "table_score": sum(table_scores) / len(table_scores),
        "formula_score": sum(formula_scores) / len(formula_scores)
    }


def omnidocbench_text_edit_distance(predictions: List[str], references: List[List[Any]], **kwargs) -> Dict[str, float]:
    """
    Calculate normalized edit distance for text.
    
    Args:
        predictions: List of model predictions
        references: List of ground truth layout detections
        
    Returns:
        Dictionary containing edit distance metrics
    """
    if len(predictions) != len(references):
        raise ValueError("Number of predictions and references must match")
    
    edit_distances = []
    
    for pred, ref in zip(predictions, references):
        # Extract text from ground truth
        gt_text = _extract_text_from_layout_dets(ref)
        
        # Calculate normalized edit distance
        edit_dist = _normalized_edit_distance(pred, gt_text)
        edit_distances.append(edit_dist)
    
    return {
        "edit_distance": sum(edit_distances) / len(edit_distances),
        "edit_distance_std": _calculate_std(edit_distances)
    }


def omnidocbench_table_teds(predictions: List[str], references: List[List[Any]], **kwargs) -> Dict[str, float]:
    """
    Calculate TEDS (Tree Edit Distance based Similarity) for tables.
    
    Args:
        predictions: List of model predictions
        references: List of ground truth layout detections
        
    Returns:
        Dictionary containing TEDS metrics
    """
    if len(predictions) != len(references):
        raise ValueError("Number of predictions and references must match")
    
    teds_scores = []
    
    for pred, ref in zip(predictions, references):
        # Extract table HTML from ground truth
        gt_table_html = _extract_table_html_from_layout_dets(ref)
        
        # Calculate TEDS score
        teds_score = _calculate_teds_score(pred, gt_table_html)
        teds_scores.append(teds_score)
    
    return {
        "teds": sum(teds_scores) / len(teds_scores),
        "teds_std": _calculate_std(teds_scores)
    }


def omnidocbench_formula_cdm(predictions: List[str], references: List[List[Any]], **kwargs) -> Dict[str, float]:
    """
    Calculate CDM (Comprehensive Distance Metric) for formulas.
    
    Args:
        predictions: List of model predictions
        references: List of ground truth layout detections
        
    Returns:
        Dictionary containing CDM metrics
    """
    if len(predictions) != len(references):
        raise ValueError("Number of predictions and references must match")
    
    cdm_scores = []
    
    for pred, ref in zip(predictions, references):
        # Extract formula LaTeX from ground truth
        gt_formula_latex = _extract_formula_latex_from_layout_dets(ref)
        
        # Calculate CDM score (simplified version)
        cdm_score = _calculate_cdm_score(pred, gt_formula_latex)
        cdm_scores.append(cdm_score)
    
    return {
        "cdm": sum(cdm_scores) / len(cdm_scores),
        "cdm_std": _calculate_std(cdm_scores)
    }


def _extract_text_from_layout_dets(layout_dets: List[Dict[str, Any]]) -> str:
    """Extract text content from layout detections."""
    text_parts = []
    for det in layout_dets:
        if det.get("category_type") in ["text_block", "title"] and det.get("text"):
            text_parts.append(det["text"])
    return " ".join(text_parts)


def _extract_table_html_from_layout_dets(layout_dets: List[Dict[str, Any]]) -> str:
    """Extract table HTML from layout detections."""
    for det in layout_dets:
        if det.get("category_type") == "table" and det.get("html"):
            return det["html"]
    return ""


def _extract_formula_latex_from_layout_dets(layout_dets: List[Dict[str, Any]]) -> str:
    """Extract formula LaTeX from layout detections."""
    for det in layout_dets:
        if det.get("category_type") == "equation_isolated" and det.get("latex"):
            return det["latex"]
    return ""


def _calculate_text_score(pred: str, ref: List[Dict[str, Any]]) -> float:
    """Calculate text score (1 - normalized edit distance)."""
    gt_text = _extract_text_from_layout_dets(ref)
    edit_dist = _normalized_edit_distance(pred, gt_text)
    return 1.0 - edit_dist


def _calculate_table_score(pred: str, ref: List[Dict[str, Any]]) -> float:
    """Calculate table TEDS score."""
    gt_table_html = _extract_table_html_from_layout_dets(ref)
    return _calculate_teds_score(pred, gt_table_html)


def _calculate_formula_score(pred: str, ref: List[Dict[str, Any]]) -> float:
    """Calculate formula CDM score."""
    gt_formula_latex = _extract_formula_latex_from_layout_dets(ref)
    return _calculate_cdm_score(pred, gt_formula_latex)


def _normalized_edit_distance(s1: str, s2: str) -> float:
    """Calculate normalized edit distance between two strings."""
    if not s1 and not s2:
        return 0.0
    if not s1 or not s2:
        return 1.0
    
    # Simple Levenshtein distance implementation
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]) + 1
    
    return dp[m][n] / max(m, n)


def _calculate_teds_score(pred: str, gt_html: str) -> float:
    """Calculate TEDS score for table evaluation."""
    if not pred or not gt_html:
        return 0.0
    
    # Simplified TEDS calculation
    # In practice, this would use the actual TEDS implementation
    pred_clean = re.sub(r'\s+', ' ', pred.strip().lower())
    gt_clean = re.sub(r'\s+', ' ', gt_html.strip().lower())
    
    if pred_clean == gt_clean:
        return 1.0
    
    # Simple similarity based on common substrings
    common_chars = sum(1 for a, b in zip(pred_clean, gt_clean) if a == b)
    max_len = max(len(pred_clean), len(gt_clean))
    
    return common_chars / max_len if max_len > 0 else 0.0


def _calculate_cdm_score(pred: str, gt_latex: str) -> float:
    """Calculate CDM score for formula evaluation."""
    if not pred or not gt_latex:
        return 0.0
    
    # Simplified CDM calculation
    # In practice, this would use the actual CDM implementation
    pred_clean = re.sub(r'\s+', ' ', pred.strip().lower())
    gt_clean = re.sub(r'\s+', ' ', gt_latex.strip().lower())
    
    if pred_clean == gt_clean:
        return 1.0
    
    # Simple similarity based on common substrings
    common_chars = sum(1 for a, b in zip(pred_clean, gt_clean) if a == b)
    max_len = max(len(pred_clean), len(gt_clean))
    
    return common_chars / max_len if max_len > 0 else 0.0


def _calculate_std(values: List[float]) -> float:
    """Calculate standard deviation of a list of values."""
    if len(values) <= 1:
        return 0.0
    
    mean_val = sum(values) / len(values)
    variance = sum((x - mean_val) ** 2 for x in values) / (len(values) - 1)
    return math.sqrt(variance)


def omnidocbench_aggregate_end2end_results(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """Aggregate OmniDocBench end-to-end results."""
    if not results:
        return {"overall_score": 0.0}
    
    predictions = [r.get("prediction", "") for r in results]
    ground_truths = [r.get("ground_truth", []) for r in results]
    
    return omnidocbench_overall_score(predictions, ground_truths)


def omnidocbench_aggregate_text_results(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """Aggregate OmniDocBench text results."""
    if not results:
        return {"edit_distance": 1.0}
    
    predictions = [r.get("prediction", "") for r in results]
    ground_truths = [r.get("ground_truth", []) for r in results]
    
    return omnidocbench_text_edit_distance(predictions, ground_truths)


def omnidocbench_aggregate_table_results(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """Aggregate OmniDocBench table results."""
    if not results:
        return {"teds": 0.0}
    
    predictions = [r.get("prediction", "") for r in results]
    ground_truths = [r.get("ground_truth", []) for r in results]
    
    return omnidocbench_table_teds(predictions, ground_truths)


def omnidocbench_aggregate_formula_results(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """Aggregate OmniDocBench formula results."""
    if not results:
        return {"cdm": 0.0}
    
    predictions = [r.get("prediction", "") for r in results]
    ground_truths = [r.get("ground_truth", []) for r in results]
    
    return omnidocbench_formula_cdm(predictions, ground_truths)


def omnidocbench_layout_map(predictions: List[List[Dict[str, Any]]], references: List[List[Dict[str, Any]]], **kwargs) -> Dict[str, float]:
    """
    Calculate mAP (mean Average Precision) for layout detection.
    
    Args:
        predictions: List of predicted bounding boxes
        references: List of ground truth layout detections
        
    Returns:
        Dictionary containing mAP metrics
    """
    if len(predictions) != len(references):
        raise ValueError("Number of predictions and references must match")
    
    # Simplified mAP calculation
    # In practice, this would use proper COCO evaluation
    total_precision = 0.0
    total_recall = 0.0
    valid_samples = 0
    
    for pred_boxes, ref_boxes in zip(predictions, references):
        if not pred_boxes or not ref_boxes:
            continue
            
        # Calculate precision and recall for this sample
        precision, recall = _calculate_precision_recall(pred_boxes, ref_boxes)
        total_precision += precision
        total_recall += recall
        valid_samples += 1
    
    if valid_samples == 0:
        return {"mAP": 0.0, "precision": 0.0, "recall": 0.0}
    
    avg_precision = total_precision / valid_samples
    avg_recall = total_recall / valid_samples
    f1_score = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall) if (avg_precision + avg_recall) > 0 else 0.0
    
    return {
        "mAP": f1_score,  # Simplified mAP
        "precision": avg_precision,
        "recall": avg_recall,
        "f1": f1_score
    }


def _calculate_precision_recall(pred_boxes: List[Dict[str, Any]], ref_boxes: List[Dict[str, Any]]) -> Tuple[float, float]:
    """Calculate precision and recall for bounding box detection."""
    if not pred_boxes or not ref_boxes:
        return 0.0, 0.0
    
    # Count matches
    matches = 0
    for pred_box in pred_boxes:
        for ref_box in ref_boxes:
            if _boxes_overlap(pred_box, ref_box) and _categories_match(pred_box, ref_box):
                matches += 1
                break
    
    precision = matches / len(pred_boxes) if pred_boxes else 0.0
    recall = matches / len(ref_boxes) if ref_boxes else 0.0
    
    return precision, recall


def _boxes_overlap(box1: Dict[str, Any], box2: Dict[str, Any], iou_threshold: float = 0.5) -> bool:
    """Check if two bounding boxes overlap based on IoU threshold."""
    try:
        # Extract coordinates
        coords1 = box1.get("poly", []) if "poly" in box1 else box1.get("bbox", [])
        coords2 = box2.get("poly", []) if "poly" in box2 else box2.get("bbox", [])
        
        if len(coords1) >= 4 and len(coords2) >= 4:
            # Convert to [x1, y1, x2, y2] format
            if len(coords1) == 8:  # Polygon format
                x1_1, y1_1, x2_1, y1_1, x2_1, y2_1, x1_1, y2_1 = coords1
                x1_1, x2_1 = min(x1_1, x2_1), max(x1_1, x2_1)
                y1_1, y2_1 = min(y1_1, y2_1), max(y1_1, y2_1)
            else:
                x1_1, y1_1, x2_1, y2_1 = coords1[:4]
            
            if len(coords2) == 8:  # Polygon format
                x1_2, y1_2, x2_2, y1_2, x2_2, y2_2, x1_2, y2_2 = coords2
                x1_2, x2_2 = min(x1_2, x2_2), max(x1_2, x2_2)
                y1_2, y2_2 = min(y1_2, y2_2), max(y1_2, y2_2)
            else:
                x1_2, y1_2, x2_2, y2_2 = coords2[:4]
            
            # Calculate IoU
            iou = _calculate_iou(x1_1, y1_1, x2_1, y2_1, x1_2, y1_2, x2_2, y2_2)
            return iou >= iou_threshold
    except:
        pass
    
    return False


def _categories_match(box1: Dict[str, Any], box2: Dict[str, Any]) -> bool:
    """Check if two bounding boxes have matching categories."""
    cat1 = box1.get("category_type", "") or box1.get("category", "")
    cat2 = box2.get("category_type", "") or box2.get("category", "")
    return cat1 == cat2


def _calculate_iou(x1_1: float, y1_1: float, x2_1: float, y2_1: float,
                   x1_2: float, y1_2: float, x2_2: float, y2_2: float) -> float:
    """Calculate Intersection over Union (IoU) of two bounding boxes."""
    # Calculate intersection
    x_left = max(x1_1, x1_2)
    y_top = max(y1_1, y1_2)
    x_right = min(x2_1, x2_2)
    y_bottom = min(y2_1, y2_2)
    
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    intersection = (x_right - x_left) * (y_bottom - y_top)
    
    # Calculate union
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0


def omnidocbench_aggregate_layout_results(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """Aggregate OmniDocBench layout detection results."""
    if not results:
        return {"mAP": 0.0}
    
    predictions = [r.get("prediction", []) for r in results]
    ground_truths = [r.get("ground_truth", []) for r in results]
    
    return omnidocbench_layout_map(predictions, ground_truths)
