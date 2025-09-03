#!/usr/bin/env python3
"""
Submission Comparison Analysis
Compares all available submissions including new BERT approaches with Phase 11 baseline
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class SubmissionComparator:
    """Comprehensive submission comparison and analysis"""
    
    def __init__(self):
        self.submissions = {}
        self.comparison_results = {}
        
    def load_all_submissions(self):
        """Load all available submissions"""
        print("Loading all available submissions...")
        
        # Load new BERT submissions
        bert_submissions = {
            'Fast BERT Features': 'submissions/fast_bert_features_submission.csv',
            'Lightning Fast Text': 'submissions/lightning_fast_text_submission.csv'
        }
        
        # Load existing submissions
        existing_submissions = {
            'Phase 11 Baseline': 'phase11_ultra_advanced_submission_20250902_215537.csv',
            'Competition Final': 'submissions/competition_final_submission.csv',
            'Advanced Ensemble': 'submissions/advanced_ensemble_submission.csv',
            'Ensemble': 'submissions/ensemble_submission.csv'
        }
        
        # Load all submissions
        all_submissions = {**bert_submissions, **existing_submissions}
        
        for name, path in all_submissions.items():
            try:
                if os.path.exists(path):
                    df = pd.read_csv(path)
                    self.submissions[name] = df
                    print(f"✓ Loaded {name}: {df.shape}")
                else:
                    print(f"✗ Not found: {path}")
            except Exception as e:
                print(f"✗ Error loading {name}: {e}")
        
        print(f"\nTotal submissions loaded: {len(self.submissions)}")
        return len(self.submissions) > 0
    
    def analyze_submission_structure(self):
        """Analyze the structure of each submission"""
        print("\n" + "="*60)
        print("SUBMISSION STRUCTURE ANALYSIS")
        print("="*60)
        
        for name, df in self.submissions.items():
            print(f"\n{name}:")
            print(f"  Shape: {df.shape}")
            print(f"  Columns: {list(df.columns)}")
            print(f"  Sample data:")
            print(f"    {df.head(3).to_string()}")
            
            # Check for missing values
            missing = df.isnull().sum()
            if missing.sum() > 0:
                print(f"  Missing values: {missing[missing > 0].to_dict()}")
            else:
                print("  No missing values")
    
    def compare_predictions(self):
        """Compare predictions across all submissions"""
        print("\n" + "="*60)
        print("PREDICTION COMPARISON ANALYSIS")
        print("="*60)
        
        # Get the first submission as reference for article IDs
        ref_name = list(self.submissions.keys())[0]
        ref_df = self.submissions[ref_name]
        
        # Create comparison dataframe
        comparison_data = []
        
        for article_id in ref_df['id']:
            row_data = {'id': article_id}
            
            for name, df in self.submissions.items():
                # Find matching row
                match = df[df['id'] == article_id]
                if not match.empty:
                    row_data[f'{name}_prediction'] = match.iloc[0]['real_text_id']
                    if 'confidence' in match.columns:
                        row_data[f'{name}_confidence'] = match.iloc[0]['confidence']
                else:
                    row_data[f'{name}_prediction'] = None
                    row_data[f'{name}_confidence'] = None
            
            comparison_data.append(row_data)
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Analyze agreement between submissions
        print("\nPrediction Agreement Analysis:")
        
        # Get prediction columns
        pred_cols = [col for col in comparison_df.columns if col.endswith('_prediction')]
        
        for i, col1 in enumerate(pred_cols):
            for j, col2 in enumerate(pred_cols[i+1:], i+1):
                name1 = col1.replace('_prediction', '')
                name2 = col2.replace('_prediction', '')
                
                # Calculate agreement
                agreement = (comparison_df[col1] == comparison_df[col2]).mean()
                print(f"  {name1} vs {name2}: {agreement:.4f} ({agreement*100:.1f}%)")
        
        # Save comparison data
        comparison_df.to_csv('submission_comparison_analysis.csv', index=False)
        print(f"\n✓ Comparison data saved to: submission_comparison_analysis.csv")
        
        return comparison_df
    
    def analyze_prediction_distribution(self):
        """Analyze prediction distributions"""
        print("\n" + "="*60)
        print("PREDICTION DISTRIBUTION ANALYSIS")
        print("="*60)
        
        for name, df in self.submissions.items():
            print(f"\n{name}:")
            
            # Count predictions
            pred_counts = df['real_text_id'].value_counts().sort_index()
            total = len(df)
            
            print(f"  Total predictions: {total}")
            print(f"  Real (1): {pred_counts.get(1, 0)} ({pred_counts.get(1, 0)/total*100:.1f}%)")
            print(f"  Fake (2): {pred_counts.get(2, 0)} ({pred_counts.get(2, 0)/total*100:.1f}%)")
            
            # Confidence analysis if available
            if 'confidence' in df.columns:
                conf_stats = df['confidence'].describe()
                print(f"  Confidence - Mean: {conf_stats['mean']:.4f}, Std: {conf_stats['std']:.4f}")
                print(f"  Confidence - Min: {conf_stats['min']:.4f}, Max: {conf_stats['max']:.4f}")
    
    def create_visualization(self):
        """Create visualizations for comparison"""
        print("\n" + "="*60)
        print("CREATING VISUALIZATIONS")
        print("="*60)
        
        try:
            # Set style
            plt.style.use('default')
            sns.set_palette("husl")
            
            # Create figure with subplots
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Submission Comparison Analysis', fontsize=16, fontweight='bold')
            
            # 1. Prediction Distribution
            ax1 = axes[0, 0]
            pred_data = []
            labels = []
            
            for name, df in self.submissions.items():
                pred_counts = df['real_text_id'].value_counts().sort_index()
                pred_data.append([pred_counts.get(1, 0), pred_counts.get(2, 0)])
                labels.append(name)
            
            pred_df = pd.DataFrame(pred_data, columns=['Real (1)', 'Fake (2)'], index=labels)
            pred_df.plot(kind='bar', ax=ax1, width=0.8)
            ax1.set_title('Prediction Distribution by Submission')
            ax1.set_ylabel('Count')
            ax1.legend()
            ax1.tick_params(axis='x', rotation=45)
            
            # 2. Confidence Distribution (if available)
            ax2 = axes[0, 1]
            conf_data = []
            conf_labels = []
            
            for name, df in self.submissions.items():
                if 'confidence' in df.columns:
                    conf_data.append(df['confidence'].values)
                    conf_labels.append(name)
            
            if conf_data:
                ax2.boxplot(conf_data, labels=conf_labels)
                ax2.set_title('Confidence Distribution by Submission')
                ax2.set_ylabel('Confidence Score')
                ax2.tick_params(axis='x', rotation=45)
            else:
                ax2.text(0.5, 0.5, 'No confidence data available', 
                        ha='center', va='center', transform=ax2.transAxes)
                ax2.set_title('Confidence Distribution (Not Available)')
            
            # 3. Agreement Matrix
            ax3 = axes[1, 0]
            agreement_matrix = []
            submission_names = list(self.submissions.keys())
            
            for i, name1 in enumerate(submission_names):
                row = []
                for j, name2 in enumerate(submission_names):
                    if i == j:
                        row.append(1.0)
                    else:
                        df1 = self.submissions[name1]
                        df2 = self.submissions[name2]
                        # Merge on ID and calculate agreement
                        merged = pd.merge(df1, df2, on='id', suffixes=('_1', '_2'))
                        agreement = (merged['real_text_id_1'] == merged['real_text_id_2']).mean()
                        row.append(agreement)
                agreement_matrix.append(row)
            
            sns.heatmap(agreement_matrix, annot=True, fmt='.3f', 
                       xticklabels=submission_names, yticklabels=submission_names,
                       cmap='RdYlBu_r', ax=ax3)
            ax3.set_title('Prediction Agreement Matrix')
            
            # 4. Summary Statistics
            ax4 = axes[1, 1]
            ax4.axis('off')
            
            # Create summary table
            summary_data = []
            for name, df in self.submissions.items():
                total = len(df)
                real_count = (df['real_text_id'] == 1).sum()
                fake_count = (df['real_text_id'] == 2).sum()
                real_pct = real_count / total * 100
                fake_pct = fake_count / total * 100
                
                summary_data.append([name, total, f"{real_count} ({real_pct:.1f}%)", 
                                   f"{fake_count} ({fake_pct:.1f}%)"])
            
            summary_df = pd.DataFrame(summary_data, 
                                    columns=['Submission', 'Total', 'Real (1)', 'Fake (2)'])
            
            table = ax4.table(cellText=summary_df.values, colLabels=summary_df.columns,
                             cellLoc='center', loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1.2, 1.5)
            ax4.set_title('Summary Statistics')
            
            # Adjust layout
            plt.tight_layout()
            
            # Save plot
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f'submission_comparison_analysis_{timestamp}.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"✓ Visualization saved to: {filename}")
            
            plt.show()
            
        except Exception as e:
            print(f"Error creating visualization: {e}")
            import traceback
            traceback.print_exc()
    
    def generate_comparison_report(self):
        """Generate comprehensive comparison report"""
        print("\n" + "="*60)
        print("GENERATING COMPARISON REPORT")
        print("="*60)
        
        report_data = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'total_submissions': len(self.submissions),
            'submission_names': list(self.submissions.keys()),
            'comparison_summary': {}
        }
        
        # Analyze each submission
        for name, df in self.submissions.items():
            total = len(df)
            real_count = (df['real_text_id'] == 1).sum()
            fake_count = (df['real_text_id'] == 2).sum()
            
            report_data['comparison_summary'][name] = {
                'total_predictions': int(total),
                'real_predictions': int(real_count),
                'fake_predictions': int(fake_count),
                'real_percentage': float(real_count / total * 100),
                'fake_percentage': float(fake_count / total * 100)
            }
        
        # Save report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f'submission_comparison_report_{timestamp}.json'
        
        with open(report_filename, 'w') as f:
            import json
            json.dump(report_data, f, indent=2)
        
        print(f"✓ Comparison report saved to: {report_filename}")
        
        # Print summary
        print("\nCOMPARISON SUMMARY:")
        print("-" * 40)
        for name, data in report_data['comparison_summary'].items():
            print(f"{name}:")
            print(f"  Total: {data['total_predictions']}")
            print(f"  Real: {data['real_predictions']} ({data['real_percentage']:.1f}%)")
            print(f"  Fake: {data['fake_predictions']} ({data['fake_percentage']:.1f}%)")
            print()
        
        return report_data
    
    def run_complete_analysis(self):
        """Run the complete comparison analysis"""
        print("="*80)
        print("SUBMISSION COMPARISON ANALYSIS")
        print("="*80)
        
        # Load submissions
        if not self.load_all_submissions():
            print("No submissions found. Analysis cannot proceed.")
            return
        
        # Run analysis
        self.analyze_submission_structure()
        self.compare_predictions()
        self.analyze_prediction_distribution()
        self.create_visualization()
        self.generate_comparison_report()
        
        print("\n" + "="*80)
        print("ANALYSIS COMPLETED SUCCESSFULLY!")
        print("="*80)

def main():
    """Main execution function"""
    comparator = SubmissionComparator()
    comparator.run_complete_analysis()

if __name__ == "__main__":
    main()
