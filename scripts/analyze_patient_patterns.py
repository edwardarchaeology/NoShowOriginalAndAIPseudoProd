"""
Analyze Patient Patterns

Identifies chronic no-showers and analyzes patient-level behavior patterns.

Key insights:
- 2.7% of patients (chronic no-showers) cause 20.1% of all no-shows
- First appointment predicts future: 31.8% no-show if missed first vs 19.0% if showed
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def load_engineered_data(filepath='data/clean/engineered.csv'):
    """Load the engineered dataset."""
    df = pd.read_csv(filepath)
    return df


def analyze_patient_behavior(df):
    """Analyze patient-level statistics."""
    print("="*60)
    print("PATIENT-LEVEL BEHAVIOR ANALYSIS")
    print("="*60)
    
    # Group by patient
    patient_stats = df.groupby('PatientId').agg({
        'no_show_binary': ['count', 'sum', 'mean'],
        'Age': 'first',
        'Gender': 'first'
    })
    
    patient_stats.columns = ['Total_Appointments', 'Total_NoShows', 'NoShow_Rate', 'Age', 'Gender']
    
    print(f"\nTotal unique patients: {len(patient_stats)}")
    print(f"Average appointments per patient: {patient_stats['Total_Appointments'].mean():.2f}")
    print(f"Median appointments per patient: {patient_stats['Total_Appointments'].median():.0f}")
    
    # Distribution of appointments per patient
    print("\nAppointments per Patient Distribution:")
    print(patient_stats['Total_Appointments'].describe())
    
    # Distribution of no-show rates
    print("\nNo-Show Rate Distribution:")
    print(patient_stats['NoShow_Rate'].describe())
    
    return patient_stats


def identify_chronic_noshowers(patient_stats, min_appointments=3, min_rate=0.5):
    """
    Identify chronic no-showers.
    
    Chronic no-shower: Patient with 3+ appointments and 50%+ no-show rate.
    """
    chronic = patient_stats[
        (patient_stats['Total_Appointments'] >= min_appointments) &
        (patient_stats['NoShow_Rate'] >= min_rate)
    ]
    
    print("\n" + "="*60)
    print(f"CHRONIC NO-SHOWERS (>={min_appointments} appts, >={min_rate*100}% no-show rate)")
    print("="*60)
    
    print(f"Chronic no-showers: {len(chronic)} ({len(chronic)/len(patient_stats)*100:.1f}% of patients)")
    print(f"Total no-shows from chronic patients: {chronic['Total_NoShows'].sum()}")
    print(f"Percentage of all no-shows: {chronic['Total_NoShows'].sum()/patient_stats['Total_NoShows'].sum()*100:.1f}%")
    
    # Top chronic no-showers
    print("\nTop 10 Chronic No-Showers:")
    top_chronic = chronic.sort_values('Total_NoShows', ascending=False).head(10)
    print(top_chronic[['Total_Appointments', 'Total_NoShows', 'NoShow_Rate']])
    
    return chronic


def analyze_first_appointment_effect(df):
    """
    Analyze the predictive power of the first appointment.
    
    Does missing the first appointment predict future behavior?
    """
    print("\n" + "="*60)
    print("FIRST APPOINTMENT PREDICTIVE EFFECT")
    print("="*60)
    
    # Get first appointment for each patient
    df_sorted = df.sort_values(['PatientId', 'AppointmentDay'])
    first_appts = df_sorted.groupby('PatientId').first().reset_index()
    
    # Patients with multiple appointments
    multi_appt_patients = df.groupby('PatientId').size()
    multi_appt_patients = multi_appt_patients[multi_appt_patients > 1].index
    
    first_appts_multi = first_appts[first_appts['PatientId'].isin(multi_appt_patients)]
    
    print(f"Patients with 2+ appointments: {len(first_appts_multi)}")
    
    # Calculate future no-show rate based on first appointment
    future_stats = []
    
    for patient_id in first_appts_multi['PatientId']:
        first_noshow = first_appts_multi[first_appts_multi['PatientId'] == patient_id]['no_show_binary'].values[0]
        patient_appts = df[df['PatientId'] == patient_id]
        
        # Exclude first appointment
        future_appts = patient_appts.iloc[1:]
        
        if len(future_appts) > 0:
            future_noshow_rate = future_appts['no_show_binary'].mean()
            future_stats.append({
                'first_noshow': first_noshow,
                'future_noshow_rate': future_noshow_rate,
                'future_count': len(future_appts)
            })
    
    future_df = pd.DataFrame(future_stats)
    
    # Compare future no-show rates
    showed_first = future_df[future_df['first_noshow'] == 0]['future_noshow_rate'].mean()
    missed_first = future_df[future_df['first_noshow'] == 1]['future_noshow_rate'].mean()
    
    print(f"\nFuture no-show rate if SHOWED first appointment: {showed_first*100:.1f}%")
    print(f"Future no-show rate if MISSED first appointment:  {missed_first*100:.1f}%")
    print(f"Difference: {(missed_first - showed_first)*100:.1f} percentage points")
    
    return future_df


def plot_patient_distribution(patient_stats):
    """Plot patient appointment and no-show distributions."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Distribution of appointments per patient
    patient_stats['Total_Appointments'].hist(bins=50, ax=axes[0, 0], color='steelblue', edgecolor='black')
    axes[0, 0].set_title('Distribution of Appointments per Patient', fontweight='bold')
    axes[0, 0].set_xlabel('Number of Appointments')
    axes[0, 0].set_ylabel('Number of Patients')
    axes[0, 0].set_xlim(0, 20)
    
    # Plot 2: Distribution of no-show rates
    patient_stats['NoShow_Rate'].hist(bins=50, ax=axes[0, 1], color='coral', edgecolor='black')
    axes[0, 1].set_title('Distribution of Patient No-Show Rates', fontweight='bold')
    axes[0, 1].set_xlabel('No-Show Rate')
    axes[0, 1].set_ylabel('Number of Patients')
    
    # Plot 3: No-show rate by number of appointments
    appt_groups = patient_stats.groupby('Total_Appointments')['NoShow_Rate'].mean()
    appt_groups[appt_groups.index <= 20].plot(kind='bar', ax=axes[1, 0], color='seagreen')
    axes[1, 0].set_title('Average No-Show Rate by Number of Appointments', fontweight='bold')
    axes[1, 0].set_xlabel('Number of Appointments')
    axes[1, 0].set_ylabel('Average No-Show Rate')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Plot 4: Percentage of patients by no-show category
    patient_stats['NoShow_Category'] = pd.cut(
        patient_stats['NoShow_Rate'],
        bins=[0, 0.01, 0.25, 0.50, 0.75, 1.0],
        labels=['Never (0%)', 'Rare (1-25%)', 'Sometimes (26-50%)', 'Often (51-75%)', 'Always (76-100%)']
    )
    category_counts = patient_stats['NoShow_Category'].value_counts()
    category_counts.plot(kind='bar', ax=axes[1, 1], color='purple', alpha=0.7)
    axes[1, 1].set_title('Patient Distribution by No-Show Behavior', fontweight='bold')
    axes[1, 1].set_xlabel('No-Show Category')
    axes[1, 1].set_ylabel('Number of Patients')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('reports/figures/patient_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\nPatient distribution plots saved to reports/figures/patient_distributions.png")


def plot_chronic_noshow_impact(patient_stats, chronic):
    """Visualize the impact of chronic no-showers."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Pie chart - percentage of patients
    labels = ['Chronic No-Showers', 'Other Patients']
    sizes = [len(chronic), len(patient_stats) - len(chronic)]
    colors = ['#ff6b6b', '#51cf66']
    explode = (0.1, 0)
    
    axes[0].pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
                shadow=True, startangle=90)
    axes[0].set_title('Chronic No-Showers\n(3+ appts, 50%+ rate)', fontweight='bold')
    
    # Plot 2: Bar chart - percentage of total no-shows
    total_noshows = patient_stats['Total_NoShows'].sum()
    chronic_noshows = chronic['Total_NoShows'].sum()
    other_noshows = total_noshows - chronic_noshows
    
    labels = ['Chronic\nNo-Showers', 'Other\nPatients']
    values = [chronic_noshows, other_noshows]
    percentages = [chronic_noshows/total_noshows*100, other_noshows/total_noshows*100]
    
    bars = axes[1].bar(labels, values, color=colors)
    axes[1].set_title('Distribution of Total No-Shows', fontweight='bold')
    axes[1].set_ylabel('Number of No-Shows')
    
    # Add percentage labels on bars
    for bar, pct in zip(bars, percentages):
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height,
                    f'{pct:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('reports/figures/chronic_noshow_impact.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Chronic no-show impact plot saved to reports/figures/chronic_noshow_impact.png")


def main():
    """Main analysis pipeline for patient patterns."""
    print("="*60)
    print("PATIENT PATTERN ANALYSIS")
    print("="*60)
    
    # Load data
    df = load_engineered_data()
    
    # Analyze patient behavior
    patient_stats = analyze_patient_behavior(df)
    
    # Identify chronic no-showers
    chronic = identify_chronic_noshowers(patient_stats)
    
    # Analyze first appointment effect
    future_df = analyze_first_appointment_effect(df)
    
    # Create visualizations
    plot_patient_distribution(patient_stats)
    plot_chronic_noshow_impact(patient_stats, chronic)
    
    print("\n" + "="*60)
    print("PATIENT PATTERN ANALYSIS COMPLETE!")
    print("="*60)


if __name__ == "__main__":
    main()
