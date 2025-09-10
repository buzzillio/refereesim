"""Command line interface for RefereeSim"""

import os
import random
import uuid
from datetime import datetime
from typing import List, Optional
import typer
from rich.console import Console
from rich.progress import track
from rich.table import Table

from .generators.paper_generator import PaperGenerator
from .seeders.error_seeder import ErrorSeeder
from .reviewers.ai_reviewer import MultiModelReviewer
from .scorers.evaluator import ReviewerEvaluator
from .utils.reporter import Reporter
from .models import StudyType, RunManifest, PaperMetadata, save_json, load_json

app = typer.Typer(help="RefereeSim: AI Reviewer Evaluation Platform")
console = Console()


@app.command()
def generate(
    num_papers: int = typer.Option(30, "--num-papers", "-n", help="Number of papers to generate"),
    output_dir: str = typer.Option("./runs", "--output", "-o", help="Output directory"),
    seed: int = typer.Option(42, "--seed", "-s", help="Random seed for reproducibility"),
    study_types: Optional[List[str]] = typer.Option(None, "--study-type", help="Study types to include"),
    num_errors: int = typer.Option(4, "--errors", "-e", help="Average number of errors per paper"),
    control_ratio: float = typer.Option(0.1, "--control-ratio", help="Ratio of papers with no errors")
):
    """Generate synthetic papers with seeded errors"""
    
    # Create run directory
    run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"
    run_dir = os.path.join(output_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)
    
    data_dir = os.path.join(run_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    
    console.print(f"[green]Starting paper generation for run: {run_id}[/green]")
    
    # Set up generators
    generator = PaperGenerator(seed)
    seeder = ErrorSeeder(seed)
    
    # Determine study types
    if study_types is None:
        available_types = [t.value for t in StudyType]
    else:
        available_types = study_types
    
    # Generate papers
    papers_metadata = []
    
    for i in track(range(num_papers), description="Generating papers..."):
        paper_id = f"paper_{i+1:03d}"
        
        # Select study type
        study_type = StudyType(random.choice(available_types))
        
        # Generate clean paper
        paper_content, metadata = generator.generate_paper(study_type, paper_id)
        
        # Decide if this should be a control paper
        is_control = random.random() < control_ratio
        
        if not is_control:
            # Seed errors
            paper_content, errors = seeder.seed_errors(paper_content, metadata, num_errors)
            metadata.errors = errors
        else:
            metadata.is_control = True
        
        papers_metadata.append(metadata)
        
        # Save paper
        paper_file = os.path.join(data_dir, f"{paper_id}.md")
        with open(paper_file, 'w') as f:
            f.write(paper_content)
        
        # Save metadata
        metadata_file = os.path.join(data_dir, f"{paper_id}_metadata.json")
        save_json(metadata, metadata_file)
    
    # Save run manifest
    manifest = RunManifest(
        run_id=run_id,
        timestamp=datetime.now().isoformat(),
        config={
            "num_papers": num_papers,
            "seed": seed,
            "study_types": available_types,
            "num_errors": num_errors,
            "control_ratio": control_ratio
        },
        seed=seed,
        papers_generated=num_papers,
        models_tested=[],
        metrics_summary={}
    )
    
    manifest_file = os.path.join(run_dir, "manifest.json")
    save_json(manifest, manifest_file)
    
    console.print(f"[green]✓ Generated {num_papers} papers in {run_dir}[/green]")
    
    # Show summary
    control_count = sum(1 for p in papers_metadata if p.is_control)
    error_count = sum(len(p.errors) for p in papers_metadata)
    
    table = Table(title="Generation Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Total Papers", str(num_papers))
    table.add_row("Control Papers", str(control_count))
    table.add_row("Papers with Errors", str(num_papers - control_count))
    table.add_row("Total Errors Seeded", str(error_count))
    table.add_row("Run Directory", run_dir)
    
    console.print(table)


@app.command()
def review(
    run_dir: str = typer.Argument(..., help="Run directory containing generated papers"),
    models: Optional[List[str]] = typer.Option(["gpt-4"], "--model", "-m", help="Models to use for review"),
    prompt_styles: Optional[List[str]] = typer.Option(["standard"], "--prompt", "-p", help="Prompt styles to test")
):
    """Run AI reviewers on generated papers"""
    
    console.print(f"[blue]Starting review process for {run_dir}[/blue]")
    
    # Load manifest
    manifest_file = os.path.join(run_dir, "manifest.json")
    if not os.path.exists(manifest_file):
        console.print("[red]Error: Run directory does not contain a valid manifest.json[/red]")
        return
    
    manifest = load_json(manifest_file, RunManifest)
    
    # Create reviews directory
    reviews_dir = os.path.join(run_dir, "reviews")
    os.makedirs(reviews_dir, exist_ok=True)
    
    # Set up reviewer
    reviewer = MultiModelReviewer(models, os.path.join(run_dir, "cache"))
    
    # Load papers
    data_dir = os.path.join(run_dir, "data")
    paper_files = [f for f in os.listdir(data_dir) if f.endswith('.md')]
    
    console.print(f"Found {len(paper_files)} papers to review")
    
    all_results = []
    
    for paper_file in track(paper_files, description="Reviewing papers..."):
        paper_id = paper_file.replace('.md', '')
        
        # Load paper content
        with open(os.path.join(data_dir, paper_file), 'r') as f:
            paper_content = f.read()
        
        # Review with all models and prompt styles
        results = reviewer.review_paper_all_models(paper_id, paper_content, prompt_styles)
        all_results.extend(results)
        
        # Save individual results
        for result in results:
            result_file = os.path.join(reviews_dir, f"{paper_id}_{result.model_name}_{result.prompt_style}.json")
            save_json(result, result_file)
    
    # Update manifest
    manifest.models_tested = models
    save_json(manifest, manifest_file)
    
    console.print(f"[green]✓ Completed {len(all_results)} reviews[/green]")
    console.print(f"Results saved in {reviews_dir}")


@app.command()
def evaluate(
    run_dir: str = typer.Argument(..., help="Run directory containing papers and reviews"),
    output_format: str = typer.Option("html", "--format", "-f", help="Output format: html, csv, or both")
):
    """Evaluate reviewer performance and generate reports"""
    
    console.print(f"[yellow]Starting evaluation for {run_dir}[/yellow]")
    
    # Load manifest
    manifest_file = os.path.join(run_dir, "manifest.json")
    manifest = load_json(manifest_file, RunManifest)
    
    # Create output directories
    scores_dir = os.path.join(run_dir, "scores")
    reports_dir = os.path.join(run_dir, "reports")
    os.makedirs(scores_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)
    
    # Load all papers and their ground truth
    data_dir = os.path.join(run_dir, "data")
    reviews_dir = os.path.join(run_dir, "reviews")
    
    evaluator = ReviewerEvaluator()
    reporter = Reporter(reports_dir)
    
    all_metrics = []
    all_ground_truth = []
    all_reviews = []
    
    # Load ground truth errors
    metadata_files = [f for f in os.listdir(data_dir) if f.endswith('_metadata.json')]
    
    for metadata_file in metadata_files:
        metadata = load_json(os.path.join(data_dir, metadata_file), PaperMetadata)
        all_ground_truth.extend(metadata.errors)
    
    # Load review results
    review_files = [f for f in os.listdir(reviews_dir) if f.endswith('.json')]
    
    for review_file in review_files:
        from ..models import ReviewResult
        review = load_json(os.path.join(reviews_dir, review_file), ReviewResult)
        all_reviews.append(review)
    
    console.print(f"Evaluating {len(all_reviews)} reviews against {len(all_ground_truth)} ground truth errors")
    
    # Evaluate each review
    for review in track(all_reviews, description="Evaluating reviews..."):
        # Find ground truth for this paper
        paper_ground_truth = []
        for metadata_file in metadata_files:
            if review.paper_id in metadata_file:
                metadata = load_json(os.path.join(data_dir, metadata_file), PaperMetadata)
                paper_ground_truth = metadata.errors
                break
        
        metrics = evaluator.evaluate_reviewer(paper_ground_truth, review)
        all_metrics.append(metrics)
    
    # Generate breakdowns
    category_breakdown = evaluator.generate_category_breakdown(all_ground_truth, all_reviews)
    difficulty_breakdown = evaluator.generate_difficulty_breakdown(all_ground_truth, all_reviews)
    
    # Update manifest with metrics summary
    manifest.metrics_summary = {m.model_name: m.model_dump() for m in all_metrics}
    save_json(manifest, manifest_file)
    
    # Generate reports
    if output_format in ["html", "both"]:
        report_file = reporter.generate_comprehensive_report(
            all_metrics, category_breakdown, difficulty_breakdown, manifest
        )
        console.print(f"[green]✓ HTML report generated: {report_file}[/green]")
        
        # Generate plots
        plot_files = reporter.generate_plots(all_metrics, category_breakdown, difficulty_breakdown)
        console.print(f"[green]✓ Generated {len(plot_files)} visualization plots[/green]")
    
    if output_format in ["csv", "both"]:
        csv_files = reporter.save_csv_data(all_metrics, category_breakdown, difficulty_breakdown)
        console.print(f"[green]✓ Generated {len(csv_files)} CSV files[/green]")
    
    # Show summary
    if all_metrics:
        avg_f1 = sum(m.f1_score for m in all_metrics) / len(all_metrics)
        best_model = max(all_metrics, key=lambda m: m.f1_score)
        
        table = Table(title="Evaluation Summary")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Average F1-Score", f"{avg_f1:.3f}")
        table.add_row("Best Model", f"{best_model.model_name} (F1: {best_model.f1_score:.3f})")
        table.add_row("Total Reviews", str(len(all_reviews)))
        table.add_row("Reports Directory", reports_dir)
        
        console.print(table)


@app.command()
def run_full_pipeline(
    num_papers: int = typer.Option(30, "--num-papers", "-n", help="Number of papers to generate"),
    models: Optional[List[str]] = typer.Option(["gpt-4"], "--model", "-m", help="Models to use for review"),
    output_dir: str = typer.Option("./runs", "--output", "-o", help="Output directory"),
    seed: int = typer.Option(42, "--seed", "-s", help="Random seed for reproducibility")
):
    """Run the complete RefereeSim pipeline: generate → review → evaluate"""
    
    console.print("[bold blue]🚀 Starting RefereeSim Full Pipeline[/bold blue]")
    
    # Step 1: Generate papers
    console.print("\n[yellow]Step 1: Generating papers...[/yellow]")
    
    # Create run directory
    run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"
    run_dir = os.path.join(output_dir, run_id)
    
    # Call generate command programmatically
    try:
        generate(num_papers=num_papers, output_dir=output_dir, seed=seed)
        console.print("[green]✓ Paper generation completed[/green]")
    except Exception as e:
        console.print(f"[red]✗ Paper generation failed: {e}[/red]")
        return
    
    # Step 2: Review papers
    console.print("\n[yellow]Step 2: Reviewing papers...[/yellow]")
    try:
        review(run_dir=run_dir, models=models)
        console.print("[green]✓ Review completed[/green]")
    except Exception as e:
        console.print(f"[red]✗ Review failed: {e}[/red]")
        return
    
    # Step 3: Evaluate and report
    console.print("\n[yellow]Step 3: Evaluating and generating reports...[/yellow]")
    try:
        evaluate(run_dir=run_dir, output_format="both")
        console.print("[green]✓ Evaluation completed[/green]")
    except Exception as e:
        console.print(f"[red]✗ Evaluation failed: {e}[/red]")
        return
    
    console.print(f"\n[bold green]🎉 RefereeSim pipeline completed successfully![/bold green]")
    console.print(f"[green]Results available in: {run_dir}[/green]")


if __name__ == "__main__":
    app()