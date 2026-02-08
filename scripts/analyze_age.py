"""
Analyze Age Patterns

Analyzes the relationship between patient age and no-show rates.

Key insight: Negative correlation - younger patients (11-20) have 25% no-show rate,
while older patients (60+) have 15% no-show rate.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def load_engineered_data(filepath='data/clean/engineered.csv'):
    """Load the engineered dataset."""
    df = pd.read_csv(filepath)
    return df


def analyze_age_distribution(df):
    """Analyze age distribution in the dataset."""
    print("="*60)
    print("AGE DISTRIBUTION ANALYSIS")
    print("="*60)
    
    print(f"\nAge Statistics:")
    print(df['Age'].describe())
    
    # Check for invalid ages
    invalid_ages = df[df['Age'] < 0]
    if len(invalid_ages) > 0:
        print(f"\nWarning: {len(invalid_ages)} records with negative age found")


def analyze_age_vs_noshow(df):
    """Analyze no-show rate by age group."""
    print("\n" + "="*60)
    print("NO-SHOW RATE BY AGE GROUP")
    print("="*60)
    
    # Create age groups
    bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 120]
    labels = ['0-10', '11-20', '21-30', '31-40', '41-50', '51-60', '61-70', '71-80', '80+']
    
    df['age_group'] = pd.cut(df['Age'], bins=bins, labels=labels)
    
    # Calculate statistics by age group
    age_stats = df.groupby('age_group', observed=True).agg({
        'no_show_binary': ['count', 'mean', 'sum']
    }).round(4)
    
    age_stats.columns = ['Count', 'No-Show Rate', 'Total No-Shows']
    
    print(age_stats)
    
    return age_stats


def plot_age_vs_noshow(df):
    """Create visualizations for age vs no-show patterns."""
    # Create age groups
    bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 120]
    labels = ['0-10', '11-20', '21-30', '31-40', '41-50', '51-60', '61-70', '71-80', '80+']
    df['age_group'] = pd.cut(df['Age'], bins=bins, labels=labels)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Age distribution
    df['Age'].hist(bins=50, ax=axes[0, 0], color='skyblue', edgecolor='black')
    axes[0, 0].set_title('Age Distribution', fontweight='bold', fontsize=12)
    axes[0, 0].set_xlabel('Age')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].axvline(df['Age'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {df["Age"].mean():.1f}')
    axes[0, 0].legend()
    
    # Plot 2: No-show rate by age group
    age_noshow = df.groupby('age_group', observed=True)['no_show_binary'].mean()
    age_noshow.plot(kind='bar', ax=axes[0, 1], color='coral')
    axes[0, 1].set_title('No-Show Rate by Age Group', fontweight='bold', fontsize=12)
    axes[0, 1].set_xlabel('Age Group')
    axes[0, 1].set_ylabel('No-Show Rate')
    axes[0, 1].axhline(df['no_show_binary'].mean(), color='red', linestyle='--', linewidth=1, alpha=0.7, label='Overall Mean')
    axes[0, 1].tick_params(axis='x', rotation=45)
    axes[0, 1].legend()
    axes[0, 1].grid(axis='y', alpha=0.3)
    
    # Plot 3: Scatter plot with trend line
    # Sample for clarity
    df_sample = df.sample(n=min(5000, len(df)), random_state=42)
    axes[1, 0].scatter(df_sample['Age'], df_sample['no_show_binary'], alpha=0.1, s=1, color='steelblue')
    
    # Add moving average
    age_bins = np.arange(0, df['Age'].max(), 5)
    moving_avg = []
    for i in range(len(age_bins)-1):
        mask = (df['Age'] >= age_bins[i]) & (df['Age'] < age_bins[i+1])
        if mask.sum() > 0:
            moving_avg.append(df[mask]['no_show_binary'].mean())
        else:
            moving_avg.append(np.nan)
    
    axes[1, 0].plot(age_bins[:-1] + 2.5, moving_avg, color='red', linewidth=3, label='5-year Moving Average')
    axes[1, 0].set_title('Age vs No-Show (with 5-year Moving Average)', fontweight='bold', fontsize=12)
    axes[1, 0].set_xlabel('Age')
    axes[1, 0].set_ylabel('No-Show (0=Show, 1=No-Show)')
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)
    
    # Plot 4: Count by age group and show status
    count_data = df.groupby(['age_group', 'no_show_binary'], observed=True).size().unstack()
    count_data.plot(kind='bar', stacked=True, ax=axes[1, 1], color=['#51cf66', '#ff6b6b'])
    axes[1, 1].set_title('Appointments by Age Group and Status', fontweight='bold', fontsize=12)
    axes[1, 1].set_xlabel('Age Group')
    axes[1, 1].set_ylabel('Number of Appointments')
    axes[1, 1].legend(['Showed', 'No-Show'])
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('reports/figures/age_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\nAge analysis plots saved to reports/figures/age_analysis.png")


def analyze_age_gender_interaction(df):
    """Analyze the interaction between age, gender, and no-show."""
    print("\n" + "="*60)
    print("AGE-GENDER INTERACTION ANALYSIS")
    print("="*60)
    
    # Create age groups
    bins = [0, 18, 30, 45, 60, 120]
    labels = ['0-18', '19-30', '31-45', '46-60', '60+']
    df['age_group_broad'] = pd.cut(df['Age'], bins=bins, labels=labels)
    
    # No-show rate by age group and gender
    interaction = df.groupby(['age_group_broad', 'Gender'], observed=True)['no_show_binary'].mean().unstack()
    
    print("\nNo-Show Rate by Age Group and Gender:")
    print(interaction)
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(10, 6))
    interaction.plot(kind='bar', ax=ax, color=['steelblue', 'coral'])
    ax.set_title('No-Show Rate by Age Group and Gender', fontweight='bold', fontsize=14)
    ax.set_xlabel('Age Group')
    ax.set_ylabel('No-Show Rate')
    ax.legend(title='Gender')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('reports/figures/age_gender_interaction.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Age-gender interaction plot saved to reports/figures/age_gender_interaction.png")


def main():
    """Main analysis pipeline for age patterns."""
    print("="*60)
    print("AGE PATTERN ANALYSIS")
    print("="*60)
    
    # Load data
    df = load_engineered_data()
    
    # Analyze age distribution
    analyze_age_distribution(df)
    
    # Analyze age vs no-show
    age_stats = analyze_age_vs_noshow(df)
    
    # Create visualizations
    plot_age_vs_noshow(df)
    
    # Analyze age-gender interaction
    analyze_age_gender_interaction(df)
    
    print("\n" + "="*60)
    print("AGE PATTERN ANALYSIS COMPLETE!")
    print("="*60)


if __name__ == "__main__":
    main()
