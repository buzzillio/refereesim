"""Evaluate reviewer performance against ground truth"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple
from collections import defaultdict
import re

from ..models import ErrorSeed, ReviewResult, ReviewFinding, EvaluationMetrics, ErrorCategory


class ReviewerEvaluator:
    """Evaluates AI reviewer performance against ground truth errors"""
    
    def __init__(self, match_threshold: float = 0.7):
        self.match_threshold = match_threshold
    
    def evaluate_reviewer(self, ground_truth_errors: List[ErrorSeed], 
                         review_result: ReviewResult) -> EvaluationMetrics:
        """Evaluate a single reviewer's performance"""
        
        # Match findings to ground truth errors
        matches = self._match_findings_to_errors(ground_truth_errors, review_result.findings)
        
        # Calculate metrics
        true_positives = len([m for m in matches if m['matched']])
        false_positives = len([f for f in review_result.findings 
                              if not any(m['finding'] == f and m['matched'] for m in matches)])
        false_negatives = len([e for e in ground_truth_errors 
                              if not any(m['error'] == e and m['matched'] for m in matches)])
        
        # Calculate standard metrics
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Additional metrics
        coverage_rate = recall  # Same as recall
        over_flag_rate = false_positives / len(review_result.findings) if len(review_result.findings) > 0 else 0
        
        return EvaluationMetrics(
            model_name=review_result.model_name,
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            true_positives=true_positives,
            false_positives=false_positives,
            false_negatives=false_negatives,
            coverage_rate=coverage_rate,
            over_flag_rate=over_flag_rate
        )
    
    def _match_findings_to_errors(self, ground_truth_errors: List[ErrorSeed], 
                                 findings: List[ReviewFinding]) -> List[Dict[str, Any]]:
        """Match reviewer findings to ground truth errors"""
        
        matches = []
        
        for finding in findings:
            best_match = None
            best_score = 0
            
            for error in ground_truth_errors:
                score = self._calculate_match_score(error, finding)
                if score > best_score and score >= self.match_threshold:
                    best_score = score
                    best_match = error
            
            matches.append({
                'finding': finding,
                'error': best_match,
                'matched': best_match is not None,
                'score': best_score
            })
        
        return matches
    
    def _calculate_match_score(self, error: ErrorSeed, finding: ReviewFinding) -> float:
        """Calculate similarity score between error and finding"""
        
        score = 0.0
        
        # Category match (40% weight)
        if finding.category.lower().replace('_', '') == error.category.value.lower().replace('_', ''):
            score += 0.4
        elif finding.category.lower() in error.category.value.lower() or error.category.value.lower() in finding.category.lower():
            score += 0.2
        
        # Location match (30% weight)
        if self._locations_match(error.location, finding.location):
            score += 0.3
        elif any(word in finding.location.lower() for word in error.location.lower().split()):
            score += 0.15
        
        # Text content match (30% weight)
        text_similarity = self._text_similarity(error.modified_text, finding.quoted_text)
        score += 0.3 * text_similarity
        
        return min(score, 1.0)
    
    def _locations_match(self, error_location: str, finding_location: str) -> bool:
        """Check if locations refer to the same place"""
        
        error_lower = error_location.lower()
        finding_lower = finding_location.lower()
        
        # Direct match
        if error_lower == finding_lower:
            return True
        
        # Check for common location indicators
        location_keywords = ['table', 'results', 'methods', 'abstract', 'discussion', 'section']
        
        for keyword in location_keywords:
            if keyword in error_lower and keyword in finding_lower:
                return True
        
        return False
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """Calculate text similarity between two strings"""
        
        if not text1 or not text2:
            return 0.0
        
        text1_lower = text1.lower()
        text2_lower = text2.lower()
        
        # Exact match
        if text1_lower == text2_lower:
            return 1.0
        
        # Check if one contains the other
        if text1_lower in text2_lower or text2_lower in text1_lower:
            return 0.8
        
        # Token overlap
        tokens1 = set(re.findall(r'\w+', text1_lower))
        tokens2 = set(re.findall(r'\w+', text2_lower))
        
        if not tokens1 or not tokens2:
            return 0.0
        
        intersection = tokens1.intersection(tokens2)
        union = tokens1.union(tokens2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def evaluate_multiple_reviewers(self, ground_truth_errors: List[ErrorSeed], 
                                   review_results: List[ReviewResult]) -> List[EvaluationMetrics]:
        """Evaluate multiple reviewers"""
        
        metrics = []
        for result in review_results:
            metric = self.evaluate_reviewer(ground_truth_errors, result)
            metrics.append(metric)
        
        return metrics
    
    def generate_category_breakdown(self, ground_truth_errors: List[ErrorSeed], 
                                   review_results: List[ReviewResult]) -> Dict[str, Any]:
        """Generate performance breakdown by error category"""
        
        category_stats = defaultdict(lambda: {
            'total_errors': 0,
            'model_performance': defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0})
        })
        
        # Count errors by category
        for error in ground_truth_errors:
            category_stats[error.category.value]['total_errors'] += 1
        
        # Evaluate each model's performance per category
        for result in review_results:
            matches = self._match_findings_to_errors(ground_truth_errors, result.findings)
            
            # Track matches by category
            for match in matches:
                if match['matched']:
                    category = match['error'].category.value
                    category_stats[category]['model_performance'][result.model_name]['tp'] += 1
                else:
                    # False positive - try to guess category from finding
                    finding_category = match['finding'].category
                    if finding_category in category_stats:
                        category_stats[finding_category]['model_performance'][result.model_name]['fp'] += 1
            
            # Count false negatives
            found_errors = set([m['error'].category.value + str(id(m['error'])) for m in matches if m['matched']])
            for error in ground_truth_errors:
                error_id = error.category.value + str(id(error))
                if error_id not in found_errors:
                    category_stats[error.category.value]['model_performance'][result.model_name]['fn'] += 1
        
        # Calculate metrics per category
        breakdown = {}
        for category, stats in category_stats.items():
            breakdown[category] = {
                'total_errors': stats['total_errors'],
                'models': {}
            }
            
            for model_name, performance in stats['model_performance'].items():
                tp = performance['tp']
                fp = performance['fp']
                fn = performance['fn']
                
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                
                breakdown[category]['models'][model_name] = {
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'true_positives': tp,
                    'false_positives': fp,
                    'false_negatives': fn
                }
        
        return breakdown
    
    def generate_difficulty_breakdown(self, ground_truth_errors: List[ErrorSeed], 
                                    review_results: List[ReviewResult]) -> Dict[str, Any]:
        """Generate performance breakdown by difficulty level"""
        
        difficulty_stats = defaultdict(lambda: {
            'total_errors': 0,
            'model_performance': defaultdict(lambda: {'found': 0, 'missed': 0})
        })
        
        # Group errors by difficulty
        errors_by_difficulty = defaultdict(list)
        for error in ground_truth_errors:
            errors_by_difficulty[error.difficulty.value].append(error)
            difficulty_stats[error.difficulty.value]['total_errors'] += 1
        
        # Evaluate performance by difficulty
        for result in review_results:
            matches = self._match_findings_to_errors(ground_truth_errors, result.findings)
            
            for difficulty, errors in errors_by_difficulty.items():
                found_errors = []
                for match in matches:
                    if match['matched'] and match['error'] in errors:
                        found_errors.append(match['error'])
                
                difficulty_stats[difficulty]['model_performance'][result.model_name]['found'] += len(found_errors)
                difficulty_stats[difficulty]['model_performance'][result.model_name]['missed'] += len(errors) - len(found_errors)
        
        # Calculate metrics
        breakdown = {}
        for difficulty, stats in difficulty_stats.items():
            breakdown[difficulty] = {
                'total_errors': stats['total_errors'],
                'models': {}
            }
            
            for model_name, performance in stats['model_performance'].items():
                found = performance['found']
                missed = performance['missed']
                total = found + missed
                
                recall = found / total if total > 0 else 0
                
                breakdown[difficulty]['models'][model_name] = {
                    'recall': recall,
                    'found': found,
                    'missed': missed,
                    'total': total
                }
        
        return breakdown