"""Inject controlled errors into papers for evaluation"""

import random
import re
import numpy as np
from typing import List, Tuple, Dict, Any, Optional

from ..models import ErrorSeed, ErrorCategory, DifficultyLevel, StudyType, PaperMetadata


class ErrorSeeder:
    """Injects controlled errors into generated papers"""
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
    
    def seed_errors(self, paper_content: str, metadata: PaperMetadata, 
                   num_errors: Optional[int] = None, difficulty_mix: Optional[Dict[str, float]] = None) -> Tuple[str, List[ErrorSeed]]:
        """Inject errors into a paper and return modified content with error list"""
        
        if num_errors is None:
            num_errors = random.randint(2, 6)
        
        if difficulty_mix is None:
            difficulty_mix = {"easy": 0.4, "medium": 0.4, "hard": 0.2}
        
        errors = []
        modified_content = paper_content
        
        # Select error types based on study type
        available_errors = self._get_available_errors(metadata.study_type)
        
        for _ in range(num_errors):
            # Select difficulty level
            difficulty = np.random.choice(
                list(difficulty_mix.keys()), 
                p=list(difficulty_mix.values())
            )
            
            # Select error category
            error_category = random.choice(available_errors)
            
            # Apply error
            error_result = self._apply_error(
                modified_content, metadata, error_category, DifficultyLevel(difficulty)
            )
            
            if error_result:
                modified_content, error_seed = error_result
                errors.append(error_seed)
        
        return modified_content, errors
    
    def _get_available_errors(self, study_type: StudyType) -> List[ErrorCategory]:
        """Get available error types for a study type"""
        
        base_errors = [
            ErrorCategory.ROUNDING_ERROR,
            ErrorCategory.TABLE_NUMBER_ERROR,
            ErrorCategory.CLAIM_CONTRADICTS_TABLE,
            ErrorCategory.FAKE_CITATION,
            ErrorCategory.SAMPLE_SIZE_MISREPORT
        ]
        
        if study_type in [StudyType.AB_TEST, StudyType.TWO_GROUP_COMPARISON, StudyType.CLINICAL_OUTCOME]:
            base_errors.extend([ErrorCategory.STATS_MISUSE])
        
        if study_type == StudyType.ML_CLASSIFICATION:
            base_errors.extend([ErrorCategory.DATA_LEAKAGE])
        
        return base_errors
    
    def _apply_error(self, content: str, metadata: PaperMetadata, 
                    category: ErrorCategory, difficulty: DifficultyLevel) -> Optional[Tuple[str, ErrorSeed]]:
        """Apply a specific error to the content"""
        
        if category == ErrorCategory.STATS_MISUSE:
            return self._apply_stats_misuse_error(content, metadata, difficulty)
        elif category == ErrorCategory.ROUNDING_ERROR:
            return self._apply_rounding_error(content, metadata, difficulty)
        elif category == ErrorCategory.TABLE_NUMBER_ERROR:
            return self._apply_table_number_error(content, metadata, difficulty)
        elif category == ErrorCategory.CLAIM_CONTRADICTS_TABLE:
            return self._apply_contradiction_error(content, metadata, difficulty)
        elif category == ErrorCategory.FAKE_CITATION:
            return self._apply_fake_citation_error(content, metadata, difficulty)
        elif category == ErrorCategory.SAMPLE_SIZE_MISREPORT:
            return self._apply_sample_size_error(content, metadata, difficulty)
        elif category == ErrorCategory.DATA_LEAKAGE:
            return self._apply_data_leakage_error(content, metadata, difficulty)
        elif category == ErrorCategory.UNIT_MISMATCH:
            return self._apply_unit_mismatch_error(content, metadata, difficulty)
        
        return None
    
    def _apply_stats_misuse_error(self, content: str, metadata: PaperMetadata, 
                                 difficulty: DifficultyLevel) -> Optional[Tuple[str, ErrorSeed]]:
        """Apply statistical misuse errors"""
        
        gt_data = metadata.ground_truth_data
        
        if metadata.study_type == StudyType.AB_TEST:
            # Wrong p-value reporting
            correct_p = gt_data["p_value"]
            
            if difficulty == DifficultyLevel.EASY:
                # Obvious wrong p-value (off by order of magnitude)
                wrong_p = correct_p * 10 if correct_p < 0.1 else correct_p / 10
                explanation = f"P-value should be {correct_p:.3f}, not {wrong_p:.3f}"
            elif difficulty == DifficultyLevel.MEDIUM:
                # Subtly wrong p-value
                wrong_p = correct_p + random.uniform(0.01, 0.05)
                explanation = f"P-value should be {correct_p:.3f}, not {wrong_p:.3f}"
            else:  # HARD
                # Wrong significance claim with borderline p-value
                if correct_p < 0.05:
                    wrong_p = correct_p + random.uniform(0.02, 0.04)  # Make it non-significant
                    explanation = f"Claims significance but p={wrong_p:.3f} > 0.05"
                else:
                    wrong_p = correct_p - random.uniform(0.02, 0.04)  # Make it significant
                    explanation = f"Claims non-significance but p={wrong_p:.3f} < 0.05"
            
            # Replace p-value in content
            p_pattern = r"p = [0-9]*\.[0-9]+"
            old_text = f"p = {correct_p:.3f}"
            new_text = f"p = {wrong_p:.3f}"
            
            if old_text in content:
                modified_content = content.replace(old_text, new_text)
                
                return modified_content, ErrorSeed(
                    category=ErrorCategory.STATS_MISUSE,
                    difficulty=difficulty,
                    location="Results section - statistical test",
                    original_text=old_text,
                    modified_text=new_text,
                    explanation=explanation,
                    confidence=0.9
                )
        
        elif metadata.study_type == StudyType.TWO_GROUP_COMPARISON:
            # Wrong t-statistic or p-value
            correct_t = gt_data["t_stat"]
            correct_p = gt_data["p_value"]
            
            if difficulty == DifficultyLevel.EASY:
                wrong_t = correct_t * 2
                explanation = f"t-statistic should be {correct_t:.3f}, not {wrong_t:.3f}"
            else:
                wrong_t = correct_t + random.uniform(0.5, 1.0)
                explanation = f"t-statistic calculation error: should be {correct_t:.3f}"
            
            old_text = f"t({gt_data['n1'] + gt_data['n2'] - 2}) = {correct_t:.3f}"
            new_text = f"t({gt_data['n1'] + gt_data['n2'] - 2}) = {wrong_t:.3f}"
            
            if old_text in content:
                modified_content = content.replace(old_text, new_text)
                
                return modified_content, ErrorSeed(
                    category=ErrorCategory.STATS_MISUSE,
                    difficulty=difficulty,
                    location="Results section - t-test",
                    original_text=old_text,
                    modified_text=new_text,
                    explanation=explanation,
                    confidence=0.85
                )
        
        return None
    
    def _apply_rounding_error(self, content: str, metadata: PaperMetadata, 
                             difficulty: DifficultyLevel) -> Optional[Tuple[str, ErrorSeed]]:
        """Apply rounding/calculation errors"""
        
        # Find percentages in tables that should sum to 100%
        table_pattern = r'\|[^|]+\|[^|]+\|[^|]+\|[^|]+\|'
        tables = re.findall(table_pattern, content, re.MULTILINE)
        
        for table_row in tables:
            # Look for percentage values
            percent_pattern = r'(\d+\.?\d*)%'
            percentages = re.findall(percent_pattern, table_row)
            
            if len(percentages) >= 2:
                # Create a sum that doesn't equal 100%
                if difficulty == DifficultyLevel.EASY:
                    # Obvious error - sum to 103% or 97%
                    error_amount = random.choice([3, -3])
                elif difficulty == DifficultyLevel.MEDIUM:
                    # Subtle error - sum to 101% or 99%
                    error_amount = random.choice([1, -1])
                else:
                    # Hard to notice - 100.5% or 99.5%
                    error_amount = random.choice([0.5, -0.5])
                
                # Modify the last percentage
                try:
                    last_percent = float(percentages[-1])
                    wrong_percent = last_percent + error_amount
                    
                    old_text = f"{last_percent}%"
                    new_text = f"{wrong_percent}%"
                    
                    if old_text in content:
                        modified_content = content.replace(old_text, new_text, 1)
                        
                        return modified_content, ErrorSeed(
                            category=ErrorCategory.ROUNDING_ERROR,
                            difficulty=difficulty,
                            location="Table - percentage calculation",
                            original_text=old_text,
                            modified_text=new_text,
                            explanation=f"Percentages don't sum to 100% due to calculation error",
                            confidence=0.8
                        )
                except ValueError:
                    continue
        
        return None
    
    def _apply_table_number_error(self, content: str, metadata: PaperMetadata, 
                                 difficulty: DifficultyLevel) -> Optional[Tuple[str, ErrorSeed]]:
        """Apply table number errors"""
        
        gt_data = metadata.ground_truth_data
        
        if metadata.study_type == StudyType.AB_TEST:
            # Wrong conversion numbers
            correct_conv_a = gt_data["conv_a"]
            correct_conv_b = gt_data["conv_b"]
            
            if difficulty == DifficultyLevel.EASY:
                wrong_conv_a = correct_conv_a + random.randint(10, 50)
            else:
                wrong_conv_a = correct_conv_a + random.randint(1, 5)
            
            old_text = f"| Control (A) | {gt_data['n_a']} | {correct_conv_a} |"
            new_text = f"| Control (A) | {gt_data['n_a']} | {wrong_conv_a} |"
            
            if old_text in content:
                modified_content = content.replace(old_text, new_text)
                
                return modified_content, ErrorSeed(
                    category=ErrorCategory.TABLE_NUMBER_ERROR,
                    difficulty=difficulty,
                    location="Table 1 - conversion count",
                    original_text=str(correct_conv_a),
                    modified_text=str(wrong_conv_a),
                    explanation=f"Conversion count should be {correct_conv_a}, not {wrong_conv_a}",
                    confidence=0.9
                )
        
        elif metadata.study_type == StudyType.ML_CLASSIFICATION:
            # Wrong confusion matrix numbers
            correct_tp = gt_data["tp"]
            
            if difficulty == DifficultyLevel.EASY:
                wrong_tp = correct_tp + random.randint(10, 30)
            else:
                wrong_tp = correct_tp + random.randint(1, 5)
            
            old_text = f"| Actual Positive | {gt_data['fn']} | {correct_tp} |"
            new_text = f"| Actual Positive | {gt_data['fn']} | {wrong_tp} |"
            
            if old_text in content:
                modified_content = content.replace(old_text, new_text)
                
                return modified_content, ErrorSeed(
                    category=ErrorCategory.TABLE_NUMBER_ERROR,
                    difficulty=difficulty,
                    location="Confusion Matrix - true positives",
                    original_text=str(correct_tp),
                    modified_text=str(wrong_tp),
                    explanation=f"True positive count should be {correct_tp}, not {wrong_tp}",
                    confidence=0.85
                )
        
        return None
    
    def _apply_contradiction_error(self, content: str, metadata: PaperMetadata, 
                                  difficulty: DifficultyLevel) -> Optional[Tuple[str, ErrorSeed]]:
        """Apply text-table contradiction errors"""
        
        gt_data = metadata.ground_truth_data
        
        if metadata.study_type == StudyType.TWO_GROUP_COMPARISON:
            # Text says one group is better but table shows opposite
            correct_mean1 = gt_data["mean1"]
            correct_mean2 = gt_data["mean2"]
            
            if correct_mean2 > correct_mean1:
                # Enhanced should be better, but text might say traditional is better
                wrong_claim = "traditional training group performed better than the enhanced training group"
                correct_claim = "enhanced training group performed better than the traditional training group"
            else:
                wrong_claim = "enhanced training group performed better than the traditional training group"
                correct_claim = "traditional training group performed better than the enhanced training group"
            
            # Find and replace claim in results section
            results_pattern = r"(The enhanced training group.*?better.*?traditional training group)"
            match = re.search(results_pattern, content, re.IGNORECASE)
            
            if match:
                old_text = match.group(1)
                new_text = wrong_claim
                modified_content = content.replace(old_text, new_text)
                
                return modified_content, ErrorSeed(
                    category=ErrorCategory.CLAIM_CONTRADICTS_TABLE,
                    difficulty=difficulty,
                    location="Results section - group comparison claim",
                    original_text=old_text,
                    modified_text=new_text,
                    explanation="Text claim contradicts the numerical results in the table",
                    confidence=0.95
                )
        
        return None
    
    def _apply_fake_citation_error(self, content: str, metadata: PaperMetadata, 
                                  difficulty: DifficultyLevel) -> Optional[Tuple[str, ErrorSeed]]:
        """Apply fake citation errors"""
        
        fake_citations = [
            "[3] Smith, J. & Doe, A. (2024). Non-existent study on statistical methods. *Fake Journal*, 999(1), 1-20.",
            "[3] Johnson, M. (2024). Imaginary research findings. *Made-up Review*, 15(5), 100-125.",
            "[3] Brown, K. et al. (2024). Fictional data analysis. *Bogus Science*, 8(3), 50-75."
        ]
        
        # Add a fake citation to the references
        fake_citation = random.choice(fake_citations)
        
        # Find the references section
        ref_pattern = r"(## References.*?)(\n\n|\Z)"
        match = re.search(ref_pattern, content, re.DOTALL)
        
        if match:
            old_refs = match.group(1)
            new_refs = old_refs + "\n\n" + fake_citation
            modified_content = content.replace(old_refs, new_refs)
            
            return modified_content, ErrorSeed(
                category=ErrorCategory.FAKE_CITATION,
                difficulty=difficulty,
                location="References section",
                original_text="[Original references]",
                modified_text=fake_citation,
                explanation="Added non-existent citation that cannot be verified",
                confidence=0.7
            )
        
        return None
    
    def _apply_sample_size_error(self, content: str, metadata: PaperMetadata, 
                                difficulty: DifficultyLevel) -> Optional[Tuple[str, ErrorSeed]]:
        """Apply sample size misreporting errors"""
        
        gt_data = metadata.ground_truth_data
        
        if metadata.study_type == StudyType.AB_TEST:
            correct_total = gt_data["n_a"] + gt_data["n_b"]
            
            if difficulty == DifficultyLevel.EASY:
                wrong_total = correct_total + random.randint(50, 100)
            else:
                wrong_total = correct_total + random.randint(5, 20)
            
            # Replace in abstract
            old_text = f"The study involved {correct_total} participants"
            new_text = f"The study involved {wrong_total} participants"
            
            if old_text in content:
                modified_content = content.replace(old_text, new_text)
                
                return modified_content, ErrorSeed(
                    category=ErrorCategory.SAMPLE_SIZE_MISREPORT,
                    difficulty=difficulty,
                    location="Abstract - sample size",
                    original_text=old_text,
                    modified_text=new_text,
                    explanation=f"Sample size should be {correct_total}, not {wrong_total}",
                    confidence=0.9
                )
        
        return None
    
    def _apply_data_leakage_error(self, content: str, metadata: PaperMetadata, 
                                 difficulty: DifficultyLevel) -> Optional[Tuple[str, ErrorSeed]]:
        """Apply data leakage errors (for ML papers)"""
        
        if metadata.study_type == StudyType.ML_CLASSIFICATION:
            # Add data leakage description
            leakage_text = "Hyperparameters were optimized using the full dataset including test data"
            correct_text = "Hyperparameters were optimized using 5-fold cross-validation on training data only"
            
            # Replace in methods section
            if correct_text in content:
                modified_content = content.replace(correct_text, leakage_text)
                
                return modified_content, ErrorSeed(
                    category=ErrorCategory.DATA_LEAKAGE,
                    difficulty=difficulty,
                    location="Methods section - model development",
                    original_text=correct_text,
                    modified_text=leakage_text,
                    explanation="Using test data for hyperparameter optimization causes data leakage",
                    confidence=0.85
                )
        
        return None
    
    def _apply_unit_mismatch_error(self, content: str, metadata: PaperMetadata, 
                                  difficulty: DifficultyLevel) -> Optional[Tuple[str, ErrorSeed]]:
        """Apply unit mismatch errors"""
        
        # Look for temperature or measurement units
        if "°C" in content:
            old_text = "measured in °C"
            new_text = "measured in °F"
            
            if old_text in content:
                modified_content = content.replace(old_text, new_text)
                
                return modified_content, ErrorSeed(
                    category=ErrorCategory.UNIT_MISMATCH,
                    difficulty=difficulty,
                    location="Methods section - measurement units",
                    original_text=old_text,
                    modified_text=new_text,
                    explanation="Unit mismatch: calculations assume Celsius but text says Fahrenheit",
                    confidence=0.8
                )
        
        return None