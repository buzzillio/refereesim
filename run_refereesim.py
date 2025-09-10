#!/usr/bin/env python3
"""
RefereeSim: AI Reviewer Evaluation Platform
Secure entry point with REAL API integration using environment variables
Run this file to get complete evaluation with confusion matrices
"""

import sys
import os
import random
import numpy as np
from datetime import datetime
import uuid

# Add current directory to path
sys.path.insert(0, '.')

from refereesim.generators.paper_generator import PaperGenerator
from refereesim.seeders.error_seeder import ErrorSeeder
from refereesim.reviewers.ai_reviewer import MultiModelReviewer
from refereesim.scorers.evaluator import ReviewerEvaluator
from refereesim.utils.reporter import Reporter
from refereesim.models import StudyType, RunManifest, save_json


def setup_environment():
    """Check for required environment variables"""
    
    # Check for required API keys in environment
    required_keys = ["COHERE_API_KEY", "HYPERBOLIC_API_KEY", "GENAI_API_KEY"]
    missing_keys = []
    
    for key in required_keys:
        if not os.getenv(key):
            missing_keys.append(key)
    
    if missing_keys:
        print(f"❌ Missing required environment variables: {', '.join(missing_keys)}")
        print("Please set these environment variables and try again.")
        print("Example:")
        for key in missing_keys:
            print(f"  export {key}=your_api_key_here")
        return False
    
    print("🔑 API keys found in environment")
    return True


def print_confusion_matrix(tp, fp, fn, tn, model_name="AI Model"):
    """Print a formatted confusion matrix to terminal"""
    
    # Focus on precision/recall/F1 which are reliable for error detection
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\n📊 PERFORMANCE METRICS - {model_name}")
    print("=" * 50)
    print("                 Predicted")
    print("               Error  No Error")
    print("Actual Error    {:4d}     {:4d}    (TP)  (FN)".format(tp, fn))
    print("   Other        {:4d}     N/A     (FP)  (-)".format(fp))
    print("-" * 50)
    print(f"Precision: {precision:.3f} ({tp}/{tp + fp})")
    print(f"Recall:    {recall:.3f} ({tp}/{tp + fn})")
    print(f"F1-Score:  {f1:.3f}")
    print("=" * 50)


def main():
    """Run the complete RefereeSim experiment with REAL API calls"""
    
    print("🚀 RefereeSim: AI Reviewer Evaluation Platform")
    print("=" * 60)
    print("✅ Using REAL API calls (not simulations)")
    print("🤖 Testing 11 working AI models (NO OpenAI required)")
    print("📊 Generating confusion matrices")
    print("=" * 60)
    
    # Setup API keys - check environment
    if not setup_environment():
        print("\n💡 Tip: You can find API keys at:")
        print("  - Cohere: https://dashboard.cohere.ai/api-keys")
        print("  - Hyperbolic: https://hyperbolic.xyz/")
        print("  - Gemini: https://ai.google.dev/gemini-api/docs/api-key")
        return
    
    # Configuration
    num_papers = 15  # Reasonable number for testing
    seed = 42
    
    # WORKING AI MODELS - REAL API INTEGRATION (NO OPENAI)
    models = [
        # Cohere Models (Real API)
        "command-a-03-2025",
        "command-r",
        
        # Gemini Models (Real API)
        "gemini-2.5-flash",
        "gemini-2.5-pro",
        
        # Hyperbolic Models (Real API)
        "openai/gpt-oss-120b",
        "meta-llama/Meta-Llama-3.1-405B-Instruct",
        "meta-llama/Meta-Llama-3.1-70B-Instruct", 
        "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "deepseek-ai/DeepSeek-R1-0528",
        "deepseek-ai/DeepSeek-R1",
        "deepseek-ai/DeepSeek-V3"
    ]
    
    print(f"🔬 Testing {len(models)} AI models with {num_papers} papers")
    
    # Create run directory
    run_id = f"refereesim_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = f"./results/{run_id}"
    os.makedirs(run_dir, exist_ok=True)
    
    data_dir = os.path.join(run_dir, "data")
    reviews_dir = os.path.join(run_dir, "reviews")
    reports_dir = os.path.join(run_dir, "reports")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(reviews_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)
    
    print(f"📁 Results will be saved to: {run_dir}")
    
    # Step 1: Generate papers with errors
    print(f"\n📝 STEP 1: Generating {num_papers} research papers with seeded errors...")
    
    generator = PaperGenerator(seed)
    seeder = ErrorSeeder(seed)
    
    papers_metadata = []
    study_types = [StudyType.AB_TEST, StudyType.TWO_GROUP_COMPARISON, StudyType.ML_CLASSIFICATION, StudyType.LINEAR_REGRESSION]
    
    for i in range(num_papers):
        paper_id = f"paper_{i+1:03d}"
        
        # Generate clean paper
        study_type = random.choice(study_types)
        paper_content, metadata = generator.generate_paper(study_type, paper_id)
        
        # Seed errors (skip 20% as controls)
        if random.random() > 0.2:
            num_errors = random.randint(2, 5)  # 2-5 errors per paper
            paper_content, errors = seeder.seed_errors(paper_content, metadata, num_errors=num_errors)
            metadata.errors = errors
        else:
            metadata.is_control = True
        
        papers_metadata.append(metadata)
        
        # Save paper
        with open(os.path.join(data_dir, f"{paper_id}.md"), 'w') as f:
            f.write(paper_content)
        
        # Save metadata
        save_json(metadata, os.path.join(data_dir, f"{paper_id}_metadata.json"))
        
        error_count = len(metadata.errors) if hasattr(metadata, 'errors') else 0
        print(f"  ✓ {paper_id} ({study_type.value}) - {error_count} errors")
    
    total_errors = sum(len(p.errors) if hasattr(p, 'errors') else 0 for p in papers_metadata)
    control_papers = sum(1 for p in papers_metadata if hasattr(p, 'is_control') and p.is_control)
    
    print(f"📊 Generated {num_papers} papers:")
    print(f"   • {total_errors} total seeded errors")
    print(f"   • {control_papers} control papers (no errors)")
    print(f"   • {num_papers - control_papers} papers with errors")
    
    # Step 2: Review papers with AI (REAL API CALLS)
    print(f"\n🤖 STEP 2: Reviewing papers with {len(models)} AI models...")
    print("⚠️  This may take several minutes due to API rate limits")
    
    reviewer = MultiModelReviewer(models, os.path.join(run_dir, "cache"))
    all_reviews = []
    
    for i, metadata in enumerate(papers_metadata):
        paper_id = metadata.paper_id
        
        # Load paper content
        with open(os.path.join(data_dir, f"{paper_id}.md"), 'r') as f:
            paper_content = f.read()
        
        print(f"\n  📄 Reviewing {paper_id} ({i+1}/{num_papers})...")
        
        try:
            # Review with all models (REAL API CALLS)
            reviews = reviewer.review_paper_all_models(paper_id, paper_content, ["standard"])
            all_reviews.extend(reviews)
            
            # Save results
            for review in reviews:
                filename = f"{paper_id}_{review.model_name.replace('/', '_')}_standard.json"
                save_json(review, os.path.join(reviews_dir, filename))
            
            print(f"    ✅ Completed {len(reviews)} model reviews")
            
        except Exception as e:
            print(f"    ❌ Error reviewing {paper_id}: {e}")
            continue
    
    print(f"\n📋 Completed {len(all_reviews)} total reviews across all models")
    
    # Step 3: Evaluate performance with CONFUSION MATRICES
    print(f"\n📊 STEP 3: Evaluating AI reviewer performance...")
    
    evaluator = ReviewerEvaluator()
    reporter = Reporter(reports_dir)
    
    all_metrics = []
    all_ground_truth = []
    
    # Collect all ground truth errors
    for metadata in papers_metadata:
        if hasattr(metadata, 'errors'):
            all_ground_truth.extend(metadata.errors)
    
    print(f"🎯 Total ground truth errors to detect: {len(all_ground_truth)}")
    
    # Evaluate each review
    model_results = {}
    
    for review in all_reviews:
        # Find ground truth for this paper
        paper_ground_truth = []
        for metadata in papers_metadata:
            if metadata.paper_id == review.paper_id:
                if hasattr(metadata, 'errors'):
                    paper_ground_truth = metadata.errors
                break
        
        metrics = evaluator.evaluate_reviewer(paper_ground_truth, review)
        all_metrics.append(metrics)
        
        # Aggregate by model
        model = review.model_name
        if model not in model_results:
            model_results[model] = {"tp": 0, "fp": 0, "fn": 0, "tn": 0, "reviews": 0}
        
        model_results[model]["tp"] += metrics.true_positives
        model_results[model]["fp"] += metrics.false_positives  
        model_results[model]["fn"] += metrics.false_negatives
        # Skip TN calculation - focus on precision/recall/F1 which are more reliable for error detection
        model_results[model]["tn"] += 0  # TN not meaningfully defined for this task
        model_results[model]["reviews"] += 1
    
    # Print confusion matrices for each model
    print(f"\n🎯 PERFORMANCE RESULTS:")
    print("=" * 80)
    
    best_model = None
    best_f1 = 0
    
    for model, results in model_results.items():
        tp, fp, fn, tn = results["tp"], results["fp"], results["fn"], results["tn"]
        
        # Calculate metrics
        total = tp + fp + fn + tn
        accuracy = (tp + tn) / total if total > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Track best model
        if f1 > best_f1:
            best_f1 = f1
            best_model = model
        
        # Print confusion matrix
        print_confusion_matrix(tp, fp, fn, tn, model)
    
    # Generate breakdowns
    category_breakdown = evaluator.generate_category_breakdown(all_ground_truth, all_reviews)
    difficulty_breakdown = evaluator.generate_difficulty_breakdown(all_ground_truth, all_reviews)
    
    # Step 4: Generate reports and save confusion matrix images
    print(f"\n📄 STEP 4: Generating comprehensive reports...")
    
    # Create manifest
    manifest = RunManifest(
        run_id=run_id,
        timestamp=datetime.now().isoformat(),
        config={"num_papers": num_papers, "models": models, "seed": seed},
        seed=seed,
        papers_generated=num_papers,
        models_tested=models,
        metrics_summary={m.model_name: m.model_dump() for m in all_metrics}
    )
    
    # Generate reports with confusion matrices
    report_file = reporter.generate_comprehensive_report(
        all_metrics, category_breakdown, difficulty_breakdown, manifest
    )
    
    plot_files = reporter.generate_plots(all_metrics, category_breakdown, difficulty_breakdown)
    csv_files = reporter.save_csv_data(all_metrics, category_breakdown, difficulty_breakdown)
    
    # Save manifest
    save_json(manifest, os.path.join(run_dir, "manifest.json"))
    
    print(f"📊 Generated HTML report: {report_file}")
    print(f"📈 Generated {len(plot_files)} visualization plots")
    print(f"🗂️ Generated confusion matrix images: confusion_matrix.png")
    print(f"📋 Generated {len(csv_files)} CSV data files")
    
    # Final summary with best model
    print(f"\n🏆 FINAL RESULTS:")
    print("=" * 60)
    print(f"📊 Papers Evaluated: {num_papers}")
    print(f"🤖 Models Tested: {len(models)}")
    print(f"🎯 Ground Truth Errors: {len(all_ground_truth)}")
    print(f"📝 Total Reviews: {len(all_reviews)}")
    
    if best_model:
        print(f"🥇 Best Model: {best_model}")
        print(f"   F1-Score: {best_f1:.3f}")
        
        # Show best model confusion matrix again
        best_results = model_results[best_model]
        print_confusion_matrix(
            best_results["tp"], best_results["fp"], 
            best_results["fn"], best_results["tn"], 
            f"🏆 WINNER: {best_model}"
        )
    
    print(f"\n✅ RefereeSim evaluation completed successfully!")
    print(f"📁 All results saved in: {run_dir}")
    print(f"🌐 Open the HTML report to see detailed analysis")
    
    return run_dir


if __name__ == "__main__":
    try:
        results_dir = main()
        print(f"\n🎉 SUCCESS! Results available in: {results_dir}")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        print("💡 Make sure you have set your API keys as environment variables")
        sys.exit(1)