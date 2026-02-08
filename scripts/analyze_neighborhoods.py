"""
Analyze Neighbourhoods

Analyzes no-show rates across different neighbourhoods.

Key insight: Wide variation across 81 neighbourhoods (0% to 100% no-show rates).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def load_engineered_data(filepath='data/clean/engineered.csv'):
    """Load the engineered dataset."""
    df = pd.read_csv(filepath)
    return df


def analyze_neighbourhood_stats(df):
    """Analyze neighbourhood-level statistics."""
    print("="*60)
    print("NEIGHBOURHOOD ANALYSIS")
    print("="*60)
    
    # Count unique neighbourhoods
    print(f"\nTotal unique neighbourhoods: {df['Neighbourhood'].nunique()}")
    
    # Calculate statistics by neighbourhood
    neighbourhood_stats = df.groupby('Neighbourhood').agg({
        'no_show_binary': ['count', 'mean', 'sum']
    }).round(4)
    
    neighbourhood_stats.columns = ['Count', 'No-Show Rate', 'Total No-Shows']
    neighbourhood_stats = neighbourhood_stats.sort_values('No-Show Rate', ascending=False)
    
    print("\nTop 10 Neighbourhoods by No-Show Rate:")
    print(neighbourhood_stats.head(10))
    
    print("\nBottom 10 Neighbourhoods by No-Show Rate:")
    print(neighbourhood_stats.tail(10))
    
    print(f"\nHighest no-show rate: {neighbourhood_stats['No-Show Rate'].max()*100:.1f}%")
    print(f"Lowest no-show rate: {neighbourhood_stats['No-Show Rate'].min()*100:.1f}%")
    print(f"Mean no-show rate: {neighbourhood_stats['No-Show Rate'].mean()*100:.1f}%")
    print(f"Std dev: {neighbourhood_stats['No-Show Rate'].std()*100:.1f}%")
    
    return neighbourhood_stats


def plot_neighbourhood_distribution(neighbourhood_stats):
    """Plot neighbourhood no-show rate distribution."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Distribution of no-show rates across neighbourhoods
    neighbourhood_stats['No-Show Rate'].hist(bins=30, ax=axes[0, 0], color='steelblue', edgecolor='black')
    axes[0, 0].set_title('Distribution of No-Show Rates Across Neighbourhoods', fontweight='bold')
    axes[0, 0].set_xlabel('No-Show Rate')
    axes[0, 0].set_ylabel('Number of Neighbourhoods')
    axes[0, 0].axvline(neighbourhood_stats['No-Show Rate'].mean(), color='red', linestyle='--', 
                       linewidth=2, label=f'Mean: {neighbourhood_stats["No-Show Rate"].mean():.3f}')
    axes[0, 0].legend()
    
    # Plot 2: Top 20 neighbourhoods by no-show rate
    top_20 = neighbourhood_stats.nlargest(20, 'No-Show Rate')
    top_20['No-Show Rate'].plot(kind='barh', ax=axes[0, 1], color='coral')
    axes[0, 1].set_title('Top 20 Neighbourhoods by No-Show Rate', fontweight='bold')
    axes[0, 1].set_xlabel('No-Show Rate')
    axes[0, 1].set_ylabel('Neighbourhood')
    axes[0, 1].invert_yaxis()
    
    # Plot 3: Scatter plot - count vs no-show rate
    axes[1, 0].scatter(neighbourhood_stats['Count'], neighbourhood_stats['No-Show Rate'], 
                      alpha=0.6, s=50, color='seagreen')
    axes[1, 0].set_title('Appointments vs No-Show Rate by Neighbourhood', fontweight='bold')
    axes[1, 0].set_xlabel('Number of Appointments')
    axes[1, 0].set_ylabel('No-Show Rate')
    axes[1, 0].set_xscale('log')
    axes[1, 0].grid(alpha=0.3)
    
    # Plot 4: Distribution of appointment counts
    neighbourhood_stats['Count'].hist(bins=50, ax=axes[1, 1], color='purple', edgecolor='black', alpha=0.7)
    axes[1, 1].set_title('Distribution of Appointments per Neighbourhood', fontweight='bold')
    axes[1, 1].set_xlabel('Number of Appointments')
    axes[1, 1].set_ylabel('Number of Neighbourhoods')
    axes[1, 1].set_yscale('log')
    
    plt.tight_layout()
    plt.savefig('reports/figures/neighbourhood_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\nNeighbourhood analysis plots saved to reports/figures/neighbourhood_analysis.png")


def analyze_top_neighbourhoods(df, neighbourhood_stats, top_n=10):
    """Analyze characteristics of top neighbourhoods."""
    print("\n" + "="*60)
    print(f"TOP {top_n} NEIGHBOURHOODS DETAILED ANALYSIS")
    print("="*60)
    
    # Get top neighbourhoods
    top_neighbourhoods = neighbourhood_stats.nlargest(top_n, 'Count').index
    
    # Filter data
    df_top = df[df['Neighbourhood'].isin(top_neighbourhoods)]
    
    print(f"\nAppointments in top {top_n} neighbourhoods: {len(df_top)} ({len(df_top)/len(df)*100:.1f}%)")
    
    # Compare characteristics
    print("\nCharacteristics of Top Neighbourhoods vs Others:")
    
    comparison_features = ['Age', 'lead_time_abs', 'SMS_received', 'medical_complexity', 'no_show_binary']
    
    for feat in comparison_features:
        if feat in df.columns:
            top_mean = df_top[feat].mean()
            other_mean = df[~df['Neighbourhood'].isin(top_neighbourhoods)][feat].mean()
            print(f"  {feat:25s}: Top={top_mean:6.2f}  Others={other_mean:6.2f}  Diff={top_mean-other_mean:6.2f}")
    
    # Create comparison plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # No-show rate for top neighbourhoods
    top_stats = neighbourhood_stats.loc[top_neighbourhoods].sort_values('No-Show Rate', ascending=False)
    
    x = range(len(top_stats))
    bars = ax.bar(x, top_stats['No-Show Rate'], color='steelblue', alpha=0.7)
    
    # Add overall mean line
    overall_mean = df['no_show_binary'].mean()
    ax.axhline(overall_mean, color='red', linestyle='--', linewidth=2, label=f'Overall Mean: {overall_mean:.3f}')
    
    ax.set_xlabel('Neighbourhood', fontsize=12)
    ax.set_ylabel('No-Show Rate', fontsize=12)
    ax.set_title(f'No-Show Rates in Top {top_n} Neighbourhoods (by Volume)', fontweight='bold', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(top_stats.index, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Color bars based on comparison to mean
    for i, (bar, rate) in enumerate(zip(bars, top_stats['No-Show Rate'])):
        if rate > overall_mean:
            bar.set_color('coral')
        else:
            bar.set_color('seagreen')
    
    plt.tight_layout()
    plt.savefig('reports/figures/top_neighbourhoods_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Top neighbourhoods comparison plot saved to reports/figures/top_neighbourhoods_comparison.png")


def main():
    """Main analysis pipeline for neighbourhood patterns."""
    print("="*60)
    print("NEIGHBOURHOOD PATTERN ANALYSIS")
    print("="*60)
    
    # Load data
    df = load_engineered_data()
    
    # Analyze neighbourhood statistics
    neighbourhood_stats = analyze_neighbourhood_stats(df)
    
    # Create visualizations
    plot_neighbourhood_distribution(neighbourhood_stats)
    
    # Analyze top neighbourhoods
    analyze_top_neighbourhoods(df, neighbourhood_stats)
    
    # Save neighbourhood stats
    neighbourhood_stats.to_csv('reports/neighbourhood_stats.csv')
    print("\nNeighbourhood statistics saved to reports/neighbourhood_stats.csv")
    
    print("\n" + "="*60)
    print("NEIGHBOURHOOD PATTERN ANALYSIS COMPLETE!")
    print("="*60)


if __name__ == "__main__":
    main()
