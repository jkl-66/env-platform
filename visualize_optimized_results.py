import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the latest result data
df = pd.read_csv('outputs/recall_comparison/traditional_vs_ai_methods_20250708_104306.csv')

# Create comparison charts
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Define colors
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8', '#F7DC6F']

# Recall rate comparison chart
bars1 = ax1.bar(range(len(df)), df['recall'], color=colors)
ax1.set_title('Recall Rate Comparison After Optimization', fontsize=14, fontweight='bold')
ax1.set_xlabel('Detection Methods')
ax1.set_ylabel('Recall Rate')
ax1.set_xticks(range(len(df)))
ax1.set_xticklabels([name.split('(')[0].strip() for name in df['method']], rotation=45, ha='right')
ax1.grid(True, alpha=0.3)

# Add value labels
for i, v in enumerate(df['recall']):
    ax1.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontweight='bold')

# F1 score comparison chart
bars2 = ax2.bar(range(len(df)), df['f1_score'], color=colors)
ax2.set_title('F1 Score Comparison After Optimization', fontsize=14, fontweight='bold')
ax2.set_xlabel('Detection Methods')
ax2.set_ylabel('F1 Score')
ax2.set_xticks(range(len(df)))
ax2.set_xticklabels([name.split('(')[0].strip() for name in df['method']], rotation=45, ha='right')
ax2.grid(True, alpha=0.3)

# Add value labels
for i, v in enumerate(df['f1_score']):
    ax2.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('outputs/recall_comparison/performance_comparison_optimized.png', dpi=300, bbox_inches='tight')
plt.show()

print("Optimized performance comparison chart has been generated and saved to: outputs/recall_comparison/performance_comparison_optimized.png")

# Print improvement summary
print("\n=== Optimization Results Summary ===")
print(f"AutoEncoder recall rate: {df[df['method'].str.contains('AutoEncoder')]['recall'].iloc[0]:.3f}")
print(f"AutoEncoder F1 score: {df[df['method'].str.contains('AutoEncoder')]['f1_score'].iloc[0]:.3f}")
print(f"Severity Threshold recall rate: {df[df['method'].str.contains('Severity')]['recall'].iloc[0]:.3f}")
print(f"Best method: {df.loc[df['f1_score'].idxmax(), 'method']} (F1: {df['f1_score'].max():.3f})")