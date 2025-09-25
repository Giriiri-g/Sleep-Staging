import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
from scipy import stats
from sklearn.decomposition import PCA
warnings.filterwarnings('ignore')

# Set up plotting parameters
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
mne.set_log_level('ERROR')

class SleepEDAAnalyzer:
    def __init__(self, data_path):
        self.data_path = Path(data_path)
        self.subjects = []
        self.raw_data = {}
        self.sleep_stages = {}
        
        # Corrected stage mapping
        self.stage_mapping = {
            1: 'Wake', 
            2: 'REM', 
            3: 'N1', 
            4: 'N2', 
            5: 'N3', 
            6: 'N4', 
            7: "Movement", 
            8: "? Not Scored"
        }
        
    def load_data(self):
        """Load PSG data and annotations for all subjects"""
        print("Loading PSG data...")
        
        epoch_files = list(self.data_path.glob("*PSG_epochs.npy"))
        
        for epoch_file in epoch_files:
            subject_id = epoch_file.name[:6]
            labels_pattern = f"{subject_id}*-PSG_labels.npy"
            matching_labels = list(self.data_path.glob(labels_pattern))
            
            if matching_labels:
                try:
                    epochs = np.load(epoch_file)
                    labels = np.load(matching_labels[0])
                    
                    self.raw_data[subject_id] = epochs
                    self.sleep_stages[subject_id] = labels
                    self.subjects.append(subject_id)
                    
                except Exception as e:
                    print(f"Error loading {subject_id}: {e}")
        
        print(f"✓ Loaded {len(self.subjects)} subjects successfully")
    
    def get_global_statistics(self):
        """Get comprehensive global statistics"""
        print("\n" + "="*60)
        print("GLOBAL SLEEP DATA ANALYSIS")
        print("="*60)
        
        # Combine all data
        all_stages = np.concatenate([self.sleep_stages[s] for s in self.subjects])
        all_epochs = np.concatenate([self.raw_data[s] for s in self.subjects])
        
        # Global statistics
        total_epochs = len(all_stages)
        total_hours = total_epochs * 30 / 3600
        
        print(f"Dataset Overview:")
        print(f"  • Total subjects: {len(self.subjects)}")
        print(f"  • Total epochs: {total_epochs:,}")
        print(f"  • Total recording time: {total_hours:.1f} hours ({total_hours/24:.1f} days)")
        print(f"  • Average recording per subject: {total_hours/len(self.subjects):.1f} hours")
        
        # Sleep stage distribution
        unique_stages, counts = np.unique(all_stages, return_counts=True)
        stage_percentages = (counts / total_epochs) * 100
        
        print(f"\nGlobal Sleep Stage Distribution:")
        for stage, count, pct in zip(unique_stages, counts, stage_percentages):
            stage_name = self.stage_mapping.get(stage, f'Stage_{stage}')
            print(f"  • {stage_name}: {count:,} epochs ({pct:.1f}%)")
        
        # Signal characteristics
        print(f"\nSignal Characteristics:")
        print(f"  • Data shape per epoch: {all_epochs.shape[1:] if len(all_epochs.shape) > 1 else 'N/A'}")
        print(f"  • Mean amplitude: {np.mean(all_epochs):.4f} ± {np.std(all_epochs):.4f}")
        print(f"  • Amplitude range: [{np.min(all_epochs):.4f}, {np.max(all_epochs):.4f}]")
        
        return {
            'all_stages': all_stages,
            'all_epochs': all_epochs,
            'stage_mapping': self.stage_mapping,
            'total_epochs': total_epochs,
            'total_hours': total_hours
        }
    
    def create_comprehensive_plots(self, data_dict):
        """Create comprehensive visualization plots"""
        all_stages = data_dict['all_stages']
        all_epochs = data_dict['all_epochs']
        stage_mapping = data_dict['stage_mapping']
        
        # Create figure with more vertical space between rows
        fig = plt.figure(figsize=(22, 20))
        gs = fig.add_gridspec(3, 2, hspace=0.5, wspace=0.35)  # Increased hspace
        
        # 1. Sleep Stage Distribution (Pie Chart)
        ax1 = fig.add_subplot(gs[0, 0])
        unique_stages, counts = np.unique(all_stages, return_counts=True)
        stage_labels = [stage_mapping.get(s, f'Stage_{s}') for s in unique_stages]
        colors = sns.color_palette("husl", len(stage_labels))
        
        wedges, texts, autotexts = ax1.pie(
            counts, labels=stage_labels, autopct='%1.1f%%', 
            colors=colors, startangle=90
        )
        ax1.set_title('Global Sleep Stage Distribution', fontsize=14, fontweight='bold')
        
        # 2. Sleep Stage Bar Chart
        ax2 = fig.add_subplot(gs[0, 1])
        bars = ax2.bar(stage_labels, counts, color=colors)
        ax2.set_title('Sleep Stage Frequencies', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Number of Epochs')
        ax2.tick_params(axis='x', rotation=45)
        
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height):,}', ha='center', va='bottom')
        
        # 3. Sleep Efficiency Metrics
        ax3 = fig.add_subplot(gs[1, 0])
        total_sleep_time = np.sum(counts[unique_stages != 0])  # All stages except wake
        sleep_efficiency = (total_sleep_time / len(all_stages)) * 100
        wake_percentage = (counts[unique_stages == 0][0] / len(all_stages)) * 100 if 0 in unique_stages else 0
        
        deep_sleep_pct = (counts[unique_stages == 4][0] / len(all_stages)) * 100 if 4 in unique_stages else 0
        rem_pct = (counts[unique_stages == 1][0] / len(all_stages)) * 100 if 1 in unique_stages else 0
        
        metrics = ['Sleep Efficiency', 'Wake %', 'Deep Sleep % (N3)', 'REM %']
        values = [sleep_efficiency, wake_percentage, deep_sleep_pct, rem_pct]
        
        bars = ax3.barh(metrics, values, color=['green', 'red', 'blue', 'purple'])
        ax3.set_title('Sleep Quality Metrics', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Percentage')
        
        # Add padding between rows
        ax3.margins(y=0.2)
        
        # 4. Subject Recording Durations
        ax4 = fig.add_subplot(gs[1, 1])
        durations = [len(self.sleep_stages[s]) * 30 / 3600 for s in self.subjects]
        ax4.hist(durations, bins=15, alpha=0.7, color='skyblue', edgecolor='black')
        ax4.set_title('Distribution of Recording Durations', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Duration (hours)')
        ax4.set_ylabel('Number of Subjects')
        ax4.axvline(np.mean(durations), color='red', linestyle='--', label=f'Mean: {np.mean(durations):.1f}h')
        ax4.legend()
        
        # 6. Sleep Architecture Analysis
        ax6 = fig.add_subplot(gs[2, 0])
        stage_transitions = []
        for i in range(len(all_stages)-1):
            if all_stages[i] != all_stages[i+1]:
                stage_transitions.append((all_stages[i], all_stages[i+1]))
        
        transition_counts = {}
        for from_stage, to_stage in stage_transitions:
            key = f"{stage_mapping.get(from_stage, from_stage)} → {stage_mapping.get(to_stage, to_stage)}"
            transition_counts[key] = transition_counts.get(key, 0) + 1
        
        top_transitions = sorted(transition_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        transition_names, transition_vals = zip(*top_transitions)
        
        bars = ax6.barh(range(len(transition_names)), transition_vals, color='lightcoral')
        ax6.set_yticks(range(len(transition_names)))
        ax6.set_yticklabels(transition_names, fontsize=10)
        ax6.set_title('Top Sleep Stage Transitions', fontsize=14, fontweight='bold')
        ax6.set_xlabel('Frequency')
        
        # Add extra spacing between rows
        ax6.margins(y=0.3)
        
        # 7. Signal Amplitude Distribution
        ax7 = fig.add_subplot(gs[2, 1])
        sample_epochs = all_epochs[::100]  
        sample_amplitudes = sample_epochs.flatten()[::1000]  
        
        ax7.hist(sample_amplitudes, bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
        ax7.set_title('Signal Amplitude Distribution', fontsize=14, fontweight='bold')
        ax7.set_xlabel('Amplitude')
        ax7.set_ylabel('Frequency')
        ax7.axvline(np.mean(sample_amplitudes), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(sample_amplitudes):.4f}')
        ax7.legend()
        
        plt.suptitle('Comprehensive Sleep EEG Data Analysis Dashboard', 
                    fontsize=18, fontweight='bold', y=0.98)
        
        plt.tight_layout()
        plt.show()
    
    def run_analysis(self):
        """Run the streamlined analysis"""
        self.load_data()
        
        if not self.subjects:
            print("No data found!")
            return None
        
        data_dict = self.get_global_statistics()
        
        print("\nGenerating comprehensive visualizations...")
        self.create_comprehensive_plots(data_dict)
        
        print("\n✓ Analysis completed!")
        return data_dict

# Usage
if __name__ == "__main__":
    data_path = r"sleep-edf-database-expanded-1.0.0\sleep-cassette\processed"
    
    analyzer = SleepEDAAnalyzer(data_path)
    results = analyzer.run_analysis()
    
    if results:
        print("\n" + "="*50)
        print("ANALYSIS SUMMARY")
        print("="*50)
        print("✓ Global sleep statistics calculated")
        print("✓ Comprehensive visualizations generated")
        print("✓ Sleep quality metrics computed")
        print("✓ Signal quality assessment completed")
