#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Optimized fusion model results visualization
Create scatter plot showing accuracy vs recall performance comparison
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import glob
from datetime import datetime

def find_latest_results():
    """Find the latest results file"""
    output_dir = Path('outputs/recall_comparison')
    csv_files = list(output_dir.glob('traditional_vs_ai_methods_*.csv'))
    
    if not csv_files:
        raise FileNotFoundError("Results file not found")
    
    # Sort by modification time, get the latest file
    latest_file = max(csv_files, key=lambda x: x.stat().st_mtime)
    return latest_file

def create_precision_recall_scatter():
    """Create precision vs recall scatter plot"""
    print("Creating scatter plot visualization for optimized fusion model...")
    
    # Read latest results
    results_file = find_latest_results()
    print(f"Reading results file: {results_file}")
    
    df = pd.read_csv(results_file)
    
    # Set figure style
    plt.style.use('default')
    plt.rcParams.update({
        'font.size': 12,
        'axes.titlesize': 16,
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 11,
        'figure.titlesize': 18
    })
    
    # Professional color scheme
    colors = {
        'traditional': '#E74C3C',      # Red
        'ai': '#3498DB',               # Blue
        'fusion': '#9B59B6',           # Purple
        'autoencoder': '#F39C12',      # Orange
        'sigma': '#27AE60',            # Green
        'grid': '#BDC3C7',             # Light gray
        'text': '#2C3E50'              # Dark gray
    }
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    fig.suptitle('Climate Anomaly Detection: Precision vs Recall Performance\n(Optimized Fusion Model Results)', 
                 fontsize=18, fontweight='bold', color=colors['text'], y=0.95)
    
    # Separate different types of methods
    traditional_methods = df[df['method_type'] == 'Traditional']
    ai_methods = df[df['method_type'] == 'AI/ML']
    
    # Further subdivide AI methods
    fusion_methods = ai_methods[ai_methods['method'].str.contains('Fusion')]
    autoencoder_methods = ai_methods[ai_methods['method'].str.contains('AutoEncoder')]
    sigma_methods = ai_methods[ai_methods['method'].str.contains('3-Sigma')]
    other_ai_methods = ai_methods[~ai_methods['method'].str.contains('Fusion|AutoEncoder|3-Sigma')]
    
    # Plot traditional methods
    if len(traditional_methods) > 0:
        scatter1 = ax.scatter(traditional_methods['precision'], traditional_methods['recall'], 
                             c=colors['traditional'], s=120, alpha=0.8, 
                             label='Traditional Methods', marker='o', 
                             edgecolors='white', linewidth=2)
    
    # Plot AutoEncoder methods
    if len(autoencoder_methods) > 0:
        scatter2 = ax.scatter(autoencoder_methods['precision'], autoencoder_methods['recall'], 
                             c=colors['autoencoder'], s=140, alpha=0.8, 
                             label='AutoEncoder', marker='s', 
                             edgecolors='white', linewidth=2)
    
    # Plot 3-Sigma methods
    if len(sigma_methods) > 0:
        scatter3 = ax.scatter(sigma_methods['precision'], sigma_methods['recall'], 
                             c=colors['sigma'], s=140, alpha=0.8, 
                             label='3-Sigma Method', marker='^', 
                             edgecolors='white', linewidth=2)
    
    # Plot fusion methods (highlighted)
    if len(fusion_methods) > 0:
        scatter4 = ax.scatter(fusion_methods['precision'], fusion_methods['recall'], 
                             c=colors['fusion'], s=250, alpha=0.9, 
                             label='Optimized Fusion Methods', marker='D', 
                             edgecolors='white', linewidth=3)
    
    # Add method labels
    for _, row in df.iterrows():
        # Use special style for fusion methods
        if 'Fusion' in row['method']:
            bbox_props = dict(boxstyle='round,pad=0.5', facecolor=colors['fusion'], alpha=0.2, edgecolor=colors['fusion'])
            fontweight = 'bold'
            fontsize = 11
        else:
            bbox_props = dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8)
            fontweight = 'normal'
            fontsize = 10
        
        # Simplify method names for display
        method_name = row['method']
        if 'Fusion_3Sigma_AE' in method_name:
            method_name = 'Optimized Fusion'
        elif 'Ensemble_Fusion' in method_name:
            method_name = 'Ensemble Fusion'
        elif 'AutoEncoder' in method_name:
            method_name = 'AutoEncoder'
        elif '3-Sigma' in method_name and 'AI/ML' in str(row['method_type']):
            method_name = '3-Sigma (AI)'
        
        ax.annotate(method_name, 
                   (row['precision'], row['recall']),
                   xytext=(8, 8), textcoords='offset points',
                   fontsize=fontsize, ha='left', va='bottom', fontweight=fontweight,
                   bbox=bbox_props)
    
    # Set axes
    ax.set_xlabel('Precision', color=colors['text'], fontsize=14, fontweight='bold')
    ax.set_ylabel('Recall', color=colors['text'], fontsize=14, fontweight='bold')
    ax.set_title('Precision vs Recall: Optimized Performance Comparison', 
                fontweight='bold', color=colors['text'], pad=20, fontsize=16)
    
    # Set axis ranges
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    
    # Add grid
    ax.grid(True, alpha=0.3, color=colors['grid'], linestyle='--')
    ax.set_facecolor('#FAFAFA')
    
    # Add performance zone identification
    # High performance zone (recall > 0.75, precision > 0.75)
    ax.axhline(y=0.75, color='green', linestyle=':', alpha=0.6, linewidth=2)
    ax.axvline(x=0.75, color='green', linestyle=':', alpha=0.6, linewidth=2)
    ax.text(0.77, 0.77, 'High Performance\nZone\n(Recall > 0.75)', fontsize=10, color='green', 
           bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.3))
    
    # Target recall line
    ax.axhline(y=0.75, color='red', linestyle='--', alpha=0.8, linewidth=2)
    ax.text(0.02, 0.76, 'Target Recall = 0.75', fontsize=10, color='red', fontweight='bold')
    
    # Add F1 contour lines
    precision_range = np.linspace(0.01, 1, 100)
    for f1_val in [0.5, 0.7, 0.8]:
        recall_line = (f1_val * precision_range) / (2 * precision_range - f1_val)
        recall_line = np.clip(recall_line, 0, 1)
        valid_mask = (recall_line >= 0) & (recall_line <= 1) & (precision_range >= f1_val/2)
        if np.any(valid_mask):
            ax.plot(precision_range[valid_mask], recall_line[valid_mask], 
                   '--', alpha=0.4, color='gray', linewidth=1)
            # Add F1 labels
            if f1_val == 0.7:
                ax.text(0.85, 0.58, f'F1={f1_val}', fontsize=9, color='gray', alpha=0.7)
            elif f1_val == 0.8:
                ax.text(0.9, 0.72, f'F1={f1_val}', fontsize=9, color='gray', alpha=0.7)
    
    # Add legend
    ax.legend(loc='lower right', frameon=True, fancybox=True, shadow=True, fontsize=11)
    
    # Add performance statistics
    stats_text = f"""Performance Summary:
• Best Recall: {df['recall'].max():.3f} ({df.loc[df['recall'].idxmax(), 'method']})
• Best F1: {df['f1_score'].max():.3f} ({df.loc[df['f1_score'].idxmax(), 'method']})
• Best Precision: {df['precision'].max():.3f} ({df.loc[df['precision'].idxmax(), 'method']})"""
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
           verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0.02, 1, 0.92])
    
    # Save chart
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path('outputs/recall_comparison')
    plot_path = output_dir / f'optimized_fusion_precision_recall_{timestamp}.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white', 
               edgecolor='none', format='png')
    print(f"Optimized fusion model scatter plot saved to: {plot_path}")
    
    # Display chart
    plt.show()
    
    # Print optimization results summary
    print("\n" + "="*80)
    print("Optimized Fusion Model Performance Summary")
    print("="*80)
    
    fusion_results = df[df['method'].str.contains('Fusion')]
    if len(fusion_results) > 0:
        print("\nFusion Model Performance:")
        for _, row in fusion_results.iterrows():
            print(f"• {row['method']}:")
            print(f"  - Recall: {row['recall']:.3f} {'✓' if row['recall'] >= 0.75 else '✗'} (Target: ≥0.75)")
            print(f"  - Precision: {row['precision']:.3f}")
            print(f"  - F1 Score: {row['f1_score']:.3f}")
            print(f"  - Accuracy: {row['accuracy']:.3f}")
            print()
    
    # Compare traditional methods and AI methods
    traditional_avg_recall = df[df['method_type'] == 'Traditional']['recall'].mean()
    ai_avg_recall = df[df['method_type'] == 'AI/ML']['recall'].mean()
    improvement = ai_avg_recall - traditional_avg_recall
    
    print(f"Performance Improvement Analysis:")
    print(f"• Traditional methods average recall: {traditional_avg_recall:.3f}")
    print(f"• AI/ML methods average recall: {ai_avg_recall:.3f}")
    print(f"• Recall improvement: +{improvement:.3f} ({improvement/traditional_avg_recall*100:+.1f}%)")
    
    return plot_path

if __name__ == "__main__":
    try:
        plot_path = create_precision_recall_scatter()
        print(f"\nVisualization completed! Chart saved to: {plot_path}")
    except Exception as e:
        print(f"Error: {e}")