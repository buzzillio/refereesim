"""Generate synthetic research papers with controlled data"""

import random
import numpy as np
from typing import Dict, Any, List, Tuple
from datetime import datetime
import scipy.stats as stats
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

from ..models import StudyType, PaperMetadata


class PaperGenerator:
    """Generates synthetic research papers with controlled ground truth data"""
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
    
    def generate_paper(self, study_type: StudyType, paper_id: str) -> Tuple[str, PaperMetadata]:
        """Generate a complete paper with metadata"""
        
        if study_type == StudyType.AB_TEST:
            return self._generate_ab_test_paper(paper_id)
        elif study_type == StudyType.TWO_GROUP_COMPARISON:
            return self._generate_two_group_paper(paper_id)
        elif study_type == StudyType.ML_CLASSIFICATION:
            return self._generate_ml_classification_paper(paper_id)
        elif study_type == StudyType.LINEAR_REGRESSION:
            return self._generate_linear_regression_paper(paper_id)
        elif study_type == StudyType.CLINICAL_OUTCOME:
            return self._generate_clinical_outcome_paper(paper_id)
        else:
            raise ValueError(f"Unknown study type: {study_type}")
    
    def _generate_ab_test_paper(self, paper_id: str) -> Tuple[str, PaperMetadata]:
        """Generate an A/B test paper"""
        
        # Generate ground truth data
        n_a = random.randint(800, 1200)
        n_b = random.randint(800, 1200)
        
        # Conversion rates
        baseline_rate = random.uniform(0.05, 0.15)
        effect_size = random.uniform(0.01, 0.03)  # Small but meaningful effect
        rate_a = baseline_rate
        rate_b = baseline_rate + effect_size
        
        # Generate conversions
        conv_a = np.random.binomial(n_a, rate_a)
        conv_b = np.random.binomial(n_b, rate_b)
        
        # Statistical test
        prop_a = conv_a / n_a
        prop_b = conv_b / n_b
        
        # Two-proportion z-test
        pooled_prop = (conv_a + conv_b) / (n_a + n_b)
        se = np.sqrt(pooled_prop * (1 - pooled_prop) * (1/n_a + 1/n_b))
        z_stat = (prop_b - prop_a) / se
        p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
        
        # Generate paper content
        title = f"Optimizing User Engagement Through Interface Design: An A/B Testing Analysis"
        
        paper_content = f"""# {title}

## Abstract

We conducted a randomized controlled experiment to evaluate the impact of a new user interface design on user engagement metrics. The study involved {n_a + n_b} participants randomly assigned to either the control group (original design, n={n_a}) or treatment group (new design, n={n_b}). Our primary outcome measure was user conversion rate. The treatment group showed a conversion rate of {prop_b:.3f} compared to {prop_a:.3f} in the control group (p={p_value:.3f}). These results suggest that the new interface design significantly improves user engagement.

## Methods

### Participants
We recruited {n_a + n_b} participants through our web platform over a 4-week period. Participants were randomly assigned to one of two conditions using a randomization algorithm.

### Design
- Control group (A): Original interface design (n={n_a})
- Treatment group (B): New interface design with enhanced visual elements (n={n_b})

### Outcome Measures
The primary outcome was binary conversion (completed desired action vs. did not complete). Secondary measures included time spent on page and click-through rates.

### Statistical Analysis
We used a two-proportion z-test to compare conversion rates between groups. Statistical significance was set at α = 0.05.

## Results

### Primary Outcome
Table 1 shows the conversion results for both groups.

| Group | N | Conversions | Conversion Rate | 95% CI |
|-------|---|-------------|-----------------|---------|
| Control (A) | {n_a} | {conv_a} | {prop_a:.3f} | [{prop_a - 1.96*np.sqrt(prop_a*(1-prop_a)/n_a):.3f}, {prop_a + 1.96*np.sqrt(prop_a*(1-prop_a)/n_a):.3f}] |
| Treatment (B) | {n_b} | {conv_b} | {prop_b:.3f} | [{prop_b - 1.96*np.sqrt(prop_b*(1-prop_b)/n_b):.3f}, {prop_b + 1.96*np.sqrt(prop_b*(1-prop_b)/n_b):.3f}] |

The difference in conversion rates was {prop_b - prop_a:.3f} (95% CI: [{(prop_b - prop_a) - 1.96*se:.3f}, {(prop_b - prop_a) + 1.96*se:.3f}]).

Statistical test results: z = {z_stat:.3f}, p = {p_value:.3f}.

## Discussion

Our findings demonstrate that the new interface design led to a statistically {'significant' if p_value < 0.05 else 'non-significant'} improvement in user conversion rates. The effect size of {((prop_b - prop_a)/prop_a)*100:.1f}% represents a meaningful business impact.

These results support the hypothesis that enhanced visual design elements can improve user engagement and conversion rates. The randomized design ensures that we can attribute the observed differences to the interface changes rather than confounding factors.

### Limitations
- The study was conducted over a limited time period
- Results may not generalize to all user populations
- We did not measure long-term retention effects

## Conclusion

The new interface design significantly improved user conversion rates compared to the original design. We recommend implementing this design change across the platform.

## References

[1] Smith, J. et al. (2023). User Interface Design and Conversion Optimization. *Journal of Web Analytics*, 15(3), 123-145.

[2] Johnson, M. & Brown, K. (2022). A/B Testing Best Practices in Digital Marketing. *Digital Marketing Review*, 8(2), 67-89.
"""

        ground_truth = {
            "n_a": n_a,
            "n_b": n_b,
            "conv_a": conv_a,
            "conv_b": conv_b,
            "prop_a": prop_a,
            "prop_b": prop_b,
            "z_stat": z_stat,
            "p_value": p_value,
            "effect_size": prop_b - prop_a,
            "pooled_prop": pooled_prop,
            "se": se
        }
        
        metadata = PaperMetadata(
            paper_id=paper_id,
            study_type=StudyType.AB_TEST,
            title=title,
            ground_truth_data=ground_truth,
            errors=[]
        )
        
        return paper_content, metadata
    
    def _generate_two_group_paper(self, paper_id: str) -> Tuple[str, PaperMetadata]:
        """Generate a two-group comparison paper"""
        
        # Generate ground truth data
        n1 = random.randint(80, 120)
        n2 = random.randint(80, 120)
        
        # Effect size and means
        baseline_mean = random.uniform(50, 70)
        effect_size = random.uniform(0.3, 0.8)  # Cohen's d
        std_dev = random.uniform(8, 15)
        
        mean1 = baseline_mean
        mean2 = baseline_mean + effect_size * std_dev
        
        # Generate data
        group1_data = np.random.normal(mean1, std_dev, n1)
        group2_data = np.random.normal(mean2, std_dev, n2)
        
        # Statistical test
        t_stat, p_value = stats.ttest_ind(group1_data, group2_data)
        
        # Calculate confidence intervals
        se1 = std_dev / np.sqrt(n1)
        se2 = std_dev / np.sqrt(n2)
        ci1_lower = mean1 - 1.96 * se1
        ci1_upper = mean1 + 1.96 * se1
        ci2_lower = mean2 - 1.96 * se2
        ci2_upper = mean2 + 1.96 * se2
        
        title = "Comparative Effectiveness of Two Training Methods on Performance Outcomes"
        
        paper_content = f"""# {title}

## Abstract

This study compared the effectiveness of two training methods on performance outcomes. Participants (N={n1 + n2}) were randomly assigned to either traditional training (n={n1}) or enhanced training (n={n2}). Performance was measured using standardized assessment scores. The enhanced training group achieved significantly higher scores (M={mean2:.2f}, SD={std_dev:.2f}) compared to the traditional training group (M={mean1:.2f}, SD={std_dev:.2f}), t({n1 + n2 - 2}) = {t_stat:.3f}, p={p_value:.3f}. These findings support the superiority of the enhanced training method.

## Methods

### Participants
{n1 + n2} participants were recruited from local organizations. Inclusion criteria included relevant background experience and availability for the full training duration.

### Design
Participants were randomly assigned to one of two training conditions:
- Traditional training group (n={n1})
- Enhanced training group (n={n2})

### Measures
Performance was assessed using a standardized assessment tool with scores ranging from 0-100. Higher scores indicate better performance.

### Statistical Analysis
Independent samples t-test was used to compare group means. Effect size was calculated using Cohen's d.

## Results

### Descriptive Statistics

| Group | N | Mean | SD | 95% CI |
|-------|---|------|----|---------| 
| Traditional | {n1} | {mean1:.2f} | {std_dev:.2f} | [{ci1_lower:.2f}, {ci1_upper:.2f}] |
| Enhanced | {n2} | {mean2:.2f} | {std_dev:.2f} | [{ci2_lower:.2f}, {ci2_upper:.2f}] |

### Inferential Statistics
The enhanced training group performed significantly better than the traditional training group, t({n1 + n2 - 2}) = {t_stat:.3f}, p = {p_value:.3f}. 

The effect size (Cohen's d) was {effect_size:.3f}, indicating a {'large' if effect_size > 0.8 else 'medium' if effect_size > 0.5 else 'small'} effect.

## Discussion

Our results demonstrate that the enhanced training method produces superior performance outcomes compared to traditional training approaches. The effect size of d = {effect_size:.3f} suggests practical significance in addition to statistical significance.

These findings have important implications for training program design and implementation. Organizations should consider adopting enhanced training methods to improve participant outcomes.

## Conclusion

Enhanced training methods significantly outperform traditional approaches, supporting their implementation in practical settings.

## References

[1] Wilson, A. et al. (2023). Training effectiveness in organizational settings. *Training & Development Journal*, 42(1), 15-32.

[2] Davis, R. & Martinez, S. (2022). Comparative training methodologies. *Educational Psychology Review*, 29(4), 445-467.
"""

        ground_truth = {
            "n1": n1,
            "n2": n2,
            "mean1": float(np.mean(group1_data)),
            "mean2": float(np.mean(group2_data)),
            "std1": float(np.std(group1_data, ddof=1)),
            "std2": float(np.std(group2_data, ddof=1)),
            "t_stat": float(t_stat),
            "p_value": float(p_value),
            "effect_size": effect_size,
            "pooled_std": std_dev
        }
        
        metadata = PaperMetadata(
            paper_id=paper_id,
            study_type=StudyType.TWO_GROUP_COMPARISON,
            title=title,
            ground_truth_data=ground_truth,
            errors=[]
        )
        
        return paper_content, metadata
    
    def _generate_ml_classification_paper(self, paper_id: str) -> Tuple[str, PaperMetadata]:
        """Generate an ML classification paper"""
        
        # Generate confusion matrix data
        n_samples = random.randint(800, 1200)
        
        # True class distribution
        class_0_ratio = random.uniform(0.4, 0.6)
        n_class_0 = int(n_samples * class_0_ratio)
        n_class_1 = n_samples - n_class_0
        
        # Model performance parameters
        sensitivity = random.uniform(0.75, 0.95)  # True positive rate
        specificity = random.uniform(0.70, 0.90)  # True negative rate
        
        # Generate confusion matrix
        tp = int(n_class_1 * sensitivity)
        fn = n_class_1 - tp
        tn = int(n_class_0 * specificity)
        fp = n_class_0 - tn
        
        # Calculate metrics
        accuracy = (tp + tn) / n_samples
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        title = "Machine Learning Approach for Automated Disease Classification"
        
        paper_content = f"""# {title}

## Abstract

We developed a machine learning classifier for automated disease classification using clinical features. The dataset consisted of {n_samples} patient records with binary classification outcomes. Our model achieved an accuracy of {accuracy:.3f}, precision of {precision:.3f}, recall of {recall:.3f}, and F1-score of {f1:.3f}. These results demonstrate the potential for automated classification in clinical decision support systems.

## Methods

### Dataset
The study used a dataset of {n_samples} patient records, with {n_class_1} positive cases and {n_class_0} negative cases. Features included demographic information, vital signs, and laboratory values.

### Model Development
We implemented a random forest classifier with 100 trees. The dataset was split into training (70%) and testing (30%) sets. Hyperparameters were optimized using 5-fold cross-validation.

### Evaluation Metrics
Model performance was assessed using accuracy, precision, recall, F1-score, and area under the ROC curve.

## Results

### Model Performance
The final model achieved the following performance on the test set:

| Metric | Value |
|--------|-------|
| Accuracy | {accuracy:.3f} |
| Precision | {precision:.3f} |
| Recall | {recall:.3f} |
| F1-Score | {f1:.3f} |

### Confusion Matrix

|              | Predicted Negative | Predicted Positive |
|--------------|-------------------|-------------------|
| Actual Negative | {tn} | {fp} |
| Actual Positive | {fn} | {tp} |

### Performance Analysis
The model correctly classified {tp + tn} out of {n_samples} cases ({accuracy*100:.1f}% accuracy). The precision of {precision:.3f} indicates that {precision*100:.1f}% of positive predictions were correct. The recall of {recall:.3f} shows that the model identified {recall*100:.1f}% of actual positive cases.

## Discussion

Our machine learning approach demonstrates promising performance for automated disease classification. The high precision minimizes false positive diagnoses, while the good recall ensures most cases are detected.

The F1-score of {f1:.3f} indicates balanced performance between precision and recall, making this model suitable for clinical deployment with appropriate oversight.

### Clinical Implications
This automated system could assist healthcare providers in screening and diagnosis, potentially reducing workload and improving consistency in classification decisions.

## Conclusion

The developed machine learning classifier shows strong performance for disease classification and warrants further validation in diverse clinical settings.

## References

[1] Chen, L. et al. (2023). Machine learning in clinical diagnosis. *Medical AI Journal*, 18(2), 89-104.

[2] Rodriguez, P. & Kim, J. (2022). Automated classification systems in healthcare. *Clinical Informatics Review*, 7(3), 234-251.
"""

        ground_truth = {
            "n_samples": n_samples,
            "n_class_0": n_class_0,
            "n_class_1": n_class_1,
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "sensitivity": sensitivity,
            "specificity": specificity
        }
        
        metadata = PaperMetadata(
            paper_id=paper_id,
            study_type=StudyType.ML_CLASSIFICATION,
            title=title,
            ground_truth_data=ground_truth,
            errors=[]
        )
        
        return paper_content, metadata
    
    def _generate_linear_regression_paper(self, paper_id: str) -> Tuple[str, PaperMetadata]:
        """Generate a linear regression paper"""
        
        # Generate regression data
        n = random.randint(100, 200)
        true_slope = random.uniform(0.5, 2.0)
        true_intercept = random.uniform(10, 30)
        noise_std = random.uniform(3, 8)
        
        # Generate predictor and outcome
        x = np.random.uniform(0, 50, n)
        y = true_intercept + true_slope * x + np.random.normal(0, noise_std, n)
        
        # Fit regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        r_squared = r_value ** 2
        
        title = "Predicting Performance Outcomes Using Continuous Predictors: A Regression Analysis"
        
        paper_content = f"""# {title}

## Abstract

This study examined the relationship between predictor variables and performance outcomes using linear regression analysis. Data from {n} participants were analyzed to test the hypothesis that predictor scores linearly predict outcome measures. Results showed a significant positive relationship (β = {slope:.3f}, p < 0.001, R² = {r_squared:.3f}), explaining {r_squared*100:.1f}% of the variance in outcomes. These findings support the predictive validity of the measure.

## Methods

### Participants
{n} participants completed both predictor and outcome assessments. Participants ranged in age from 18 to 65 years.

### Measures
- Predictor variable: Continuous scale (0-50)
- Outcome variable: Performance score (continuous)

### Statistical Analysis
Simple linear regression was used to examine the relationship between predictor and outcome variables. Assumptions of linearity, normality, and homoscedasticity were tested.

## Results

### Descriptive Statistics
- Predictor: M = {np.mean(x):.2f}, SD = {np.std(x):.2f}
- Outcome: M = {np.mean(y):.2f}, SD = {np.std(y):.2f}

### Regression Analysis
The regression equation was: Outcome = {intercept:.3f} + {slope:.3f} × Predictor

| Parameter | Estimate | SE | t | p |
|-----------|----------|----|----|---|
| Intercept | {intercept:.3f} | {std_err:.3f} | {intercept/std_err:.2f} | < 0.001 |
| Slope | {slope:.3f} | {std_err:.3f} | {slope/std_err:.2f} | < 0.001 |

Model fit: R² = {r_squared:.3f}, F(1,{n-2}) = {(slope/std_err)**2:.2f}, p < 0.001

### Effect Size
The correlation coefficient was r = {r_value:.3f}, indicating a {'strong' if abs(r_value) > 0.7 else 'moderate' if abs(r_value) > 0.3 else 'weak'} positive relationship.

## Discussion

Our analysis revealed a significant positive relationship between the predictor and outcome variables. The R² value of {r_squared:.3f} indicates that {r_squared*100:.1f}% of the variance in performance outcomes can be explained by the predictor variable.

This relationship supports the theoretical framework suggesting that predictor scores are meaningful indicators of performance potential. The effect size is practically significant for predictive applications.

### Implications
These findings support the use of the predictor measure in selection and assessment contexts. The linear relationship facilitates straightforward prediction of outcomes.

## Conclusion

The predictor variable demonstrates strong predictive validity for performance outcomes, supporting its practical application.

## References

[1] Thompson, K. et al. (2023). Predictive modeling in behavioral research. *Statistical Methods Journal*, 31(4), 178-195.

[2] Anderson, M. & White, C. (2022). Regression analysis in psychological measurement. *Measurement Science*, 15(2), 67-84.
"""

        ground_truth = {
            "n": n,
            "slope": float(slope),
            "intercept": float(intercept),
            "r_value": float(r_value),
            "r_squared": float(r_squared),
            "p_value": float(p_value),
            "std_err": float(std_err),
            "mean_x": float(np.mean(x)),
            "mean_y": float(np.mean(y)),
            "std_x": float(np.std(x)),
            "std_y": float(np.std(y)),
            "true_slope": true_slope,
            "true_intercept": true_intercept
        }
        
        metadata = PaperMetadata(
            paper_id=paper_id,
            study_type=StudyType.LINEAR_REGRESSION,
            title=title,
            ground_truth_data=ground_truth,
            errors=[]
        )
        
        return paper_content, metadata
    
    def _generate_clinical_outcome_paper(self, paper_id: str) -> Tuple[str, PaperMetadata]:
        """Generate a clinical outcome paper"""
        
        # Generate 2x2 contingency table data
        n_treatment = random.randint(150, 250)
        n_control = random.randint(150, 250)
        
        # Event rates
        control_event_rate = random.uniform(0.15, 0.35)
        relative_risk_reduction = random.uniform(0.2, 0.4)
        treatment_event_rate = control_event_rate * (1 - relative_risk_reduction)
        
        # Generate events
        events_control = np.random.binomial(n_control, control_event_rate)
        events_treatment = np.random.binomial(n_treatment, treatment_event_rate)
        
        # Calculate outcomes
        no_events_control = n_control - events_control
        no_events_treatment = n_treatment - events_treatment
        
        # Statistical analysis
        contingency_table = np.array([[events_treatment, no_events_treatment],
                                    [events_control, no_events_control]])
        
        chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
        
        # Risk measures
        risk_treatment = events_treatment / n_treatment
        risk_control = events_control / n_control
        risk_ratio = risk_treatment / risk_control if risk_control > 0 else 0
        risk_difference = risk_treatment - risk_control
        
        # Odds ratio
        odds_treatment = events_treatment / no_events_treatment if no_events_treatment > 0 else float('inf')
        odds_control = events_control / no_events_control if no_events_control > 0 else float('inf')
        odds_ratio = odds_treatment / odds_control if odds_control > 0 and odds_control != float('inf') else 0
        
        title = "Randomized Controlled Trial of Novel Treatment for Clinical Outcomes"
        
        paper_content = f"""# {title}

## Abstract

We conducted a randomized controlled trial to evaluate the effectiveness of a novel treatment on clinical outcomes. Participants (N={n_treatment + n_control}) were randomly assigned to treatment (n={n_treatment}) or control (n={n_control}) groups. The primary endpoint was occurrence of adverse events within 30 days. The treatment group had {events_treatment} events ({risk_treatment*100:.1f}%) compared to {events_control} events ({risk_control*100:.1f}%) in the control group (RR = {risk_ratio:.3f}, 95% CI, p = {p_value:.3f}). These results demonstrate significant efficacy of the novel treatment.

## Methods

### Study Design
Double-blind, randomized controlled trial conducted at multiple clinical sites.

### Participants
{n_treatment + n_control} patients meeting inclusion criteria were enrolled and randomized 1:1 to treatment or control groups.

### Intervention
- Treatment group: Novel therapeutic intervention (n={n_treatment})
- Control group: Standard care (n={n_control})

### Primary Outcome
Occurrence of adverse events within 30 days of randomization.

### Statistical Analysis
Chi-square test was used to compare event rates between groups. Risk ratios and odds ratios were calculated with 95% confidence intervals.

## Results

### Baseline Characteristics
Groups were well-balanced on baseline demographic and clinical characteristics.

### Primary Outcome Analysis

| Group | N | Events | Event Rate | 95% CI |
|-------|---|--------|------------|---------|
| Treatment | {n_treatment} | {events_treatment} | {risk_treatment:.3f} | [{risk_treatment - 1.96*np.sqrt(risk_treatment*(1-risk_treatment)/n_treatment):.3f}, {risk_treatment + 1.96*np.sqrt(risk_treatment*(1-risk_treatment)/n_treatment):.3f}] |
| Control | {n_control} | {events_control} | {risk_control:.3f} | [{risk_control - 1.96*np.sqrt(risk_control*(1-risk_control)/n_control):.3f}, {risk_control + 1.96*np.sqrt(risk_control*(1-risk_control)/n_control):.3f}] |

### Statistical Comparison
- Risk Ratio: {risk_ratio:.3f}
- Risk Difference: {risk_difference:.3f}
- Odds Ratio: {odds_ratio:.3f}
- Chi-square: χ²(1) = {chi2:.3f}, p = {p_value:.3f}

The treatment significantly reduced the risk of adverse events compared to control (p {'< 0.05' if p_value < 0.05 else '≥ 0.05'}).

## Discussion

Our trial demonstrates that the novel treatment significantly reduces adverse event rates compared to standard care. The relative risk reduction of {(1-risk_ratio)*100:.1f}% represents a clinically meaningful benefit.

The number needed to treat (NNT) is {1/abs(risk_difference):.1f}, meaning approximately {1/abs(risk_difference):.0f} patients need to be treated to prevent one adverse event.

### Clinical Implications
These findings support the adoption of the novel treatment as standard care for this patient population. The favorable risk-benefit profile justifies clinical implementation.

## Conclusion

The novel treatment significantly reduces adverse events and should be considered for standard clinical practice.

## References

[1] Miller, S. et al. (2023). Clinical trial methodology in therapeutic research. *Clinical Trials Journal*, 25(3), 145-162.

[2] Johnson, R. & Lee, H. (2022). Risk assessment in randomized trials. *Medical Statistics Review*, 12(4), 289-307.
"""

        ground_truth = {
            "n_treatment": n_treatment,
            "n_control": n_control,
            "events_treatment": events_treatment,
            "events_control": events_control,
            "risk_treatment": float(risk_treatment),
            "risk_control": float(risk_control),
            "risk_ratio": float(risk_ratio),
            "odds_ratio": float(odds_ratio),
            "risk_difference": float(risk_difference),
            "chi2": float(chi2),
            "p_value": float(p_value),
            "control_event_rate": control_event_rate,
            "treatment_event_rate": treatment_event_rate
        }
        
        metadata = PaperMetadata(
            paper_id=paper_id,
            study_type=StudyType.CLINICAL_OUTCOME,
            title=title,
            ground_truth_data=ground_truth,
            errors=[]
        )
        
        return paper_content, metadata