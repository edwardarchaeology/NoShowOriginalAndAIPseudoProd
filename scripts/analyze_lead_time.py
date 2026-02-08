"""
Analyze Lead Time Patterns

Creates the U-shaped curve visualization showing the relationship
between lead time and no-show rate.

Key insight: Same-day appointments have the LOWEST no-show rate (4.7%),
while appointments 2-4 weeks out have the HIGHEST rate (32.6%).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def load_engineered_data(filepath='data/clean/engineered.csv'):
    """Load the engineered dataset."""
    df = pd.read_csv(filepath)
    return df


def analyze_lead_time_distribution(df):
    """Analyze the distribution of lead times."""
    print("Lead Time Distribution:")
    print(f"Mean: {df['lead_time_days'].mean():.2f} days")
    print(f"Median: {df['lead_time_days'].median():.2f} days")
    print(f"Min: {df['lead_time_days'].min():.2f} days")
    print(f"Max: {df['lead_time_days'].max():.2f} days")
    print(f"\nSame-day or past appointments: {(df['is_same_day_or_past']==1).sum()} ({(df['is_same_day_or_past']==1).sum()/len(df)*100:.1f}%)")


def analyze_lead_time_vs_noshow(df):
    """Analyze no-show rate by lead time category."""
    print("\n" + "="*60)
    print("No-Show Rate by Lead Time Category")
    print("="*60)
    
    # Group by lead time category
    category_stats = df.groupby('lead_time_category').agg({
        'no_show_binary': ['count', 'mean', 'sum']
    }).round(4)
    
    category_stats.columns = ['Count', 'No-Show Rate', 'Total No-Shows']
    category_stats = category_stats.sort_values('No-Show Rate')
    
    print(category_stats)
    
    return category_stats


def plot_lead_time_curve(df):
    """
    Create the U-shaped curve visualization.
    
    This is the CRITICAL visualization showing that same-day appointments
    are protective (lowest risk), while appointments weeks out are highest risk.
    """
    # Bin lead time for visualization
    bins = [-np.inf, 0, 1, 7, 14, 21, 30, 60, np.inf]
    labels = ['Same/Past', '1 day', '1 week', '2 weeks', '3 weeks', '1 month', '2 months', '>2 months']
    
    df['lead_time_bin'] = pd.cut(df['lead_time_days'], bins=bins, labels=labels)
    
    # Calculate no-show rate by bin
    bin_stats = df.groupby('lead_time_bin', observed=True).agg({
        'no_show_binary': ['count', 'mean']
    })
    bin_stats.columns = ['Count', 'No-Show Rate']
    bin_stats = bin_stats.reset_index()
    
    # Create plot
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # Bar plot for count
    ax1.bar(range(len(bin_stats)), bin_stats['Count'], alpha=0.3, color='steelblue', label='Count')
    ax1.set_xlabel('Lead Time', fontsize=12)
    ax1.set_ylabel('Number of Appointments', fontsize=12, color='steelblue')
    ax1.tick_params(axis='y', labelcolor='steelblue')
    ax1.set_xticks(range(len(bin_stats)))
    ax1.set_xticklabels(bin_stats['lead_time_bin'], rotation=45, ha='right')
    
    # Line plot for no-show rate
    ax2 = ax1.twinx()
    ax2.plot(range(len(bin_stats)), bin_stats['No-Show Rate'], 
             color='red', marker='o', linewidth=2, markersize=8, label='No-Show Rate')
    ax2.set_ylabel('No-Show Rate', fontsize=12, color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    ax2.axhline(y=df['no_show_binary'].mean(), color='orange', linestyle='--', 
                linewidth=1, alpha=0.7, label='Overall Mean')
    
    # Title and legend
    plt.title('U-Shaped Relationship: Lead Time vs No-Show Rate', 
              fontsize=14, fontweight='bold')
    
    # Add annotations for key insights
    same_day_rate = bin_stats.iloc[0]['No-Show Rate']
    max_idx = bin_stats['No-Show Rate'].idxmax()
    max_rate = bin_stats.iloc[max_idx]['No-Show Rate']
    
    ax2.annotate(f'Lowest: {same_day_rate*100:.1f}%', 
                xy=(0, same_day_rate), xytext=(0.5, same_day_rate-0.05),
                arrowprops=dict(arrowstyle='->', color='green', lw=2),
                fontsize=10, fontweight='bold', color='green')
    
    ax2.annotate(f'Highest: {max_rate*100:.1f}%', 
                xy=(max_idx, max_rate), xytext=(max_idx+0.5, max_rate+0.05),
                arrowprops=dict(arrowstyle='->', color='darkred', lw=2),
                fontsize=10, fontweight='bold', color='darkred')
    
    fig.tight_layout()
    plt.savefig('reports/figures/lead_time_u_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\nU-shaped curve saved to reports/figures/lead_time_u_curve.png")
    print(f"Same-day no-show rate: {same_day_rate*100:.1f}%")
    print(f"Highest no-show rate: {max_rate*100:.1f}% (at {bin_stats.iloc[max_idx]['lead_time_bin']})")


def analyze_sms_vs_lead_time(df):
    """
    Analyze the relationship between SMS and lead time.
    
    Shows that SMS is confounded with lead time (sent only for longer lead times).
    """
    print("\n" + "="*60)
    print("SMS Reminder vs Lead Time Analysis")
    print("="*60)
    
    # SMS by lead time category
    sms_stats = df.groupby(['lead_time_category', 'SMS_received']).size().unstack(fill_value=0)
    sms_stats['SMS_Rate'] = sms_stats[1] / (sms_stats[0] + sms_stats[1])
    
    print("\nSMS Rate by Lead Time Category:")
    print(sms_stats[['SMS_Rate']])
    
    # No-show rate by SMS status
    print("\nNo-Show Rate by SMS Status:")
    print(df.groupby('SMS_received')['no_show_binary'].agg(['count', 'mean']))
    
    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: SMS rate by lead time
    sms_stats['SMS_Rate'].plot(kind='bar', ax=axes[0], color='skyblue')
    axes[0].set_title('SMS Reminder Rate by Lead Time', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Lead Time Category')
    axes[0].set_ylabel('SMS Rate')
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].grid(axis='y', alpha=0.3)
    
    # Plot 2: No-show rate by SMS and lead time
    cross_tab = df.groupby(['lead_time_category', 'SMS_received'])['no_show_binary'].mean().unstack()
    cross_tab.plot(kind='bar', ax=axes[1], color=['coral', 'steelblue'])
    axes[1].set_title('No-Show Rate by SMS Status and Lead Time', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Lead Time Category')
    axes[1].set_ylabel('No-Show Rate')
    axes[1].legend(['No SMS', 'SMS Received'])
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('reports/figures/sms_vs_lead_time.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\nSMS analysis plot saved to reports/figures/sms_vs_lead_time.png")


def main():
    """Main analysis pipeline for lead time patterns."""
    print("="*60)
    print("LEAD TIME ANALYSIS")
    print("="*60)
    
    # Load data
    df = load_engineered_data()
    
    # Analyze lead time distribution
    analyze_lead_time_distribution(df)
    
    # Analyze lead time vs no-show
    category_stats = analyze_lead_time_vs_noshow(df)
    
    # Create U-shaped curve visualization
    plot_lead_time_curve(df)
    
    # Analyze SMS vs lead time
    analyze_sms_vs_lead_time(df)
    
    print("\n" + "="*60)
    print("LEAD TIME ANALYSIS COMPLETE!")
    print("="*60)


if __name__ == "__main__":
    main()
