#!/usr/bin/env python3
"""
Visualize metrics from all inference runs in inference_outputs/inference/
Creates a comprehensive matplotlib figure with all available metrics plots.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from collections import defaultdict


def load_test_summaries(inference_dir):
    """Load all test_summary.json files from inference runs."""
    inference_path = Path(inference_dir)
    summaries = {}
    
    for run_dir in inference_path.iterdir():
        if run_dir.is_dir():
            summary_file = run_dir / "test_summary.json"
            if summary_file.exists():
                with open(summary_file, 'r') as f:
                    summaries[run_dir.name] = json.load(f)
                print(f"Loaded: {run_dir.name}")
    
    return summaries


def extract_metric_series(summaries):
    """Extract time series data for each metric across all runs."""
    # Structure: {run_name: {metric_name: {step: value}}}
    run_metrics = {}
    
    for run_name, summary in summaries.items():
        run_metrics[run_name] = defaultdict(dict)
        
        # Get a sample scene to extract metrics
        if "per_scene" in summary and summary["per_scene"]:
            # Average across all scenes
            all_scenes = summary["per_scene"]
            
            # Collect all metric names and steps
            sample_scene = next(iter(all_scenes.values()))
            
            for metric_name in sample_scene.keys():
                if isinstance(sample_scene[metric_name], dict):
                    # This is a time series metric (has init, opt_1, opt_2, etc.)
                    steps = sample_scene[metric_name].keys()
                    
                    for step in steps:
                        # Average across all scenes for this step
                        values = []
                        for scene_data in all_scenes.values():
                            if metric_name in scene_data and step in scene_data[metric_name]:
                                values.append(scene_data[metric_name][step])
                        
                        if values:
                            run_metrics[run_name][metric_name][step] = np.mean(values)
    
    return run_metrics


def parse_step_name(step):
    """Convert step name to numeric value for plotting."""
    if step == "init":
        return 0
    elif step.startswith("opt_"):
        return int(step.split("_")[1])
    elif step == "ft":
        return 500  # Put finetune at x=500 to create visual gap
    else:
        return float('inf')


def plot_all_metrics(run_metrics, output_path):
    """Create comprehensive figure with all metrics."""
    # Collect all unique metrics
    all_metrics = set()
    for run_data in run_metrics.values():
        all_metrics.update(run_data.keys())
    
    all_metrics = sorted(all_metrics)
    
    # Determine grid size
    n_metrics = len(all_metrics)
    n_cols = 3
    n_rows = (n_metrics + n_cols - 1) // n_cols
    
    # Create figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5*n_rows))
    if n_metrics == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    
    # Color map for different runs
    colors = plt.cm.tab10(np.linspace(0, 1, len(run_metrics)))
    
    # Plot each metric
    for idx, metric_name in enumerate(all_metrics):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]
        # First pass: determine max_opt_step for this metric
        max_opt_step = 0
        has_ft = False
        for run_name, metrics in run_metrics.items():
            if metric_name in metrics:
                for step in metrics[metric_name].keys():
                    if step == "ft":
                        has_ft = True
                    elif step.startswith("opt_"):
                        max_opt_step = max(max_opt_step, int(step.split("_")[1]))
        
        # Plot each run
        for run_idx, (run_name, metrics) in enumerate(run_metrics.items()):
            if metric_name in metrics:
                # Sort steps
                steps_data = metrics[metric_name]
                sorted_steps = sorted(steps_data.items(), key=lambda x: parse_step_name(x[0]))
                
                steps = [s[0] for s in sorted_steps]
                values = [s[1] for s in sorted_steps]
                
                # Convert step names to numeric for x-axis
                x_values = [parse_step_name(s) for s in steps]
                
                # Remap ft step from 500 to max_opt_step + 1 for visual display
                x_values_remapped = []
                for x, s in zip(x_values, steps):
                    if s == "ft":
                        x_values_remapped.append(max_opt_step + 1)
                    else:
                        x_values_remapped.append(x)
                
                # Separate opt steps and ft step
                opt_steps = [(x, v) for x, v, s in zip(x_values_remapped, values, steps) if s != "ft"]
                ft_step = [(x, v) for x, v, s in zip(x_values_remapped, values, steps) if s == "ft"]
                
                if opt_steps:
                    opt_x = [x[0] for x in opt_steps]
                    opt_v = [x[1] for x in opt_steps]
                    # Plot init and opt steps as connected line
                    ax.plot(opt_x, opt_v, 
                           marker='o', label=run_name, color=colors[run_idx],
                           linewidth=2, markersize=4)
                
                # Only plot ft step for "phase2_eval" run
                if ft_step and run_name == "phase2_eval":
                    # Plot ft step as a separate marker at remapped position
                    ax.scatter([ft_step[0][0]], [ft_step[0][1]], 
                              color=colors[run_idx], marker='*', s=200, 
                              edgecolors='black', linewidth=1, zorder=5)
                    
                    # Connect last opt step to ft step with dotted line
                    if opt_steps:
                        last_opt_x = opt_steps[-1][0]
                        last_opt_v = opt_steps[-1][1]
                        ft_x = ft_step[0][0]
                        ft_v = ft_step[0][1]
                        ax.plot([last_opt_x, ft_x], [last_opt_v, ft_v],
                               linestyle='--', color=colors[run_idx], 
                               linewidth=1.5, alpha=0.7)
        
        # Set custom x-axis ticks
        # Check if any run has ft step for this metric (already computed above)
        
        if has_ft:
            # Set x-axis limits to show all steps including remapped ft
            ax.set_xlim(-0.5, max_opt_step + 1.5)
            
            # Create custom tick positions and labels
            tick_positions = []
            tick_labels = []
            
            # Add init
            tick_positions.append(0)
            tick_labels.append('0')
            
            # Add opt steps (show every 5th if too many)
            if max_opt_step <= 10:
                opt_ticks = list(range(1, max_opt_step + 1))
            else:
                opt_ticks = [i for i in range(5, max_opt_step + 1, 5)]
            
            for t in opt_ticks:
                tick_positions.append(t)
                tick_labels.append(str(t))
            
            # Add ft at remapped position with label "500"
            tick_positions.append(max_opt_step + 1)
            tick_labels.append('500')
            
            ax.set_xticks(tick_positions)
            ax.set_xticklabels(tick_labels, fontsize=8)
        
        # Format metric name for title
        title = metric_name.replace("_", " ").title()
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('Optimization Step', fontsize=10)
        ax.set_ylabel('Value', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc='best')
        
        # Special handling for time metric (log scale can be useful)
        if metric_name == "time":
            ax.set_ylabel('Time (seconds)', fontsize=10)
    
    # Hide empty subplots
    for idx in range(n_metrics, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].set_visible(False)
    
    plt.suptitle('Inference Metrics Comparison Across Runs', 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    # Save figure
    output_path = Path(output_path)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved figure to: {output_path}")
    
    return fig


def main():
    inference_dir = "/home/geiger/gwb987/work/codebase/QuickSplat/inference_outputs/inference"
    output_path = "/home/geiger/gwb987/work/codebase/QuickSplat/inference_outputs/metrics_comparison.png"
    
    print("Loading test summaries...")
    summaries = load_test_summaries(inference_dir)
    
    if not summaries:
        print("No test_summary.json files found!")
        return
    
    print(f"\nFound {len(summaries)} runs")
    
    print("\nExtracting metrics...")
    run_metrics = extract_metric_series(summaries)
    
    print("\nCreating plots...")
    plot_all_metrics(run_metrics, output_path)
    
    print("\nDone!")
    
    # Print summary
    print("\n" + "="*60)
    print("Summary:")
    print("="*60)
    for run_name, metrics in run_metrics.items():
        print(f"\n{run_name}:")
        print(f"  Metrics: {', '.join(sorted(metrics.keys()))}")
        # Print number of optimization steps
        if metrics:
            sample_metric = next(iter(metrics.values()))
            n_opt_steps = len([k for k in sample_metric.keys() if k.startswith("opt_")])
            print(f"  Optimization steps: {n_opt_steps}")


if __name__ == "__main__":
    main()
