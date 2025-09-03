#!/usr/bin/env python3
"""
Immediate Competition Submission: Fast BERT Features
Submits Fast BERT Features submission to test competition performance
"""

import os
import pandas as pd
import shutil
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class FastBERTCompetitionSubmitter:
    """Immediate competition submission for Fast BERT Features"""
    
    def __init__(self):
        self.submission_file = 'submissions/fast_bert_features_submission.csv'
        self.competition_file = 'fast_bert_competition_submission.csv'
        self.backup_dir = 'competition_submissions'
        
    def validate_submission(self):
        """Validate the submission file before sending"""
        print("="*60)
        print("VALIDATING FAST BERT SUBMISSION")
        print("="*60)
        
        if not os.path.exists(self.submission_file):
            print(f"✗ Submission file not found: {self.submission_file}")
            return False
        
        # Load and validate submission
        try:
            df = pd.read_csv(self.submission_file)
            print(f"✓ Submission loaded: {df.shape}")
            
            # Check required columns
            required_cols = ['id', 'real_text_id']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                print(f"✗ Missing required columns: {missing_cols}")
                return False
            
            # Check data types and ranges
            print(f"  ID range: {df['id'].min()} to {df['id'].max()}")
            print(f"  Real text ID values: {sorted(df['real_text_id'].unique())}")
            
            # Validate real_text_id values (should be 1 or 2)
            invalid_values = df[~df['real_text_id'].isin([1, 2])]
            if not invalid_values.empty:
                print(f"✗ Invalid real_text_id values found: {invalid_values['real_text_id'].unique()}")
                return False
            
            # Check for missing values
            missing_data = df.isnull().sum()
            if missing_data.sum() > 0:
                print(f"✗ Missing values found: {missing_data[missing_data > 0].to_dict()}")
                return False
            
            print("✓ Submission validation passed!")
            return True
            
        except Exception as e:
            print(f"✗ Error validating submission: {e}")
            return False
    
    def prepare_competition_submission(self):
        """Prepare the submission for competition"""
        print("\n" + "="*60)
        print("PREPARING COMPETITION SUBMISSION")
        print("="*60)
        
        try:
            # Load submission
            df = pd.read_csv(self.submission_file)
            
            # Create competition-ready version (remove confidence column if needed)
            if 'confidence' in df.columns:
                competition_df = df[['id', 'real_text_id']].copy()
                print("  Removed confidence column for competition format")
            else:
                competition_df = df.copy()
            
            # Ensure proper data types
            competition_df['id'] = competition_df['id'].astype(int)
            competition_df['real_text_id'] = competition_df['real_text_id'].astype(int)
            
            # Save competition version
            competition_df.to_csv(self.competition_file, index=False)
            print(f"✓ Competition submission saved: {self.competition_file}")
            
            # Create backup
            os.makedirs(self.backup_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_file = f"{self.backup_dir}/fast_bert_competition_submission_{timestamp}.csv"
            competition_df.to_csv(backup_file, index=False)
            print(f"✓ Backup created: {backup_file}")
            
            return competition_df
            
        except Exception as e:
            print(f"✗ Error preparing competition submission: {e}")
            return None
    
    def generate_submission_summary(self, df):
        """Generate submission summary for tracking"""
        print("\n" + "="*60)
        print("SUBMISSION SUMMARY")
        print("="*60)
        
        total = len(df)
        real_count = (df['real_text_id'] == 1).sum()
        fake_count = (df['real_text_id'] == 2).sum()
        
        print(f"Total predictions: {total}")
        print(f"Real (1): {real_count} ({real_count/total*100:.1f}%)")
        print(f"Fake (2): {fake_count} ({fake_count/total*100:.1f}%)")
        
        # Create submission tracking file
        tracking_data = {
            'submission_name': 'Fast BERT Features',
            'submission_timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'total_predictions': int(total),
            'real_predictions': int(real_count),
            'fake_predictions': int(fake_count),
            'real_percentage': float(real_count/total*100),
            'fake_percentage': float(fake_count/total*100),
            'competition_file': self.competition_file,
            'status': 'Ready for submission',
            'notes': 'BERT-based feature extraction with Logistic Regression classifier'
        }
        
        # Save tracking info
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        tracking_file = f"{self.backup_dir}/fast_bert_submission_tracking_{timestamp}.json"
        import json
        with open(tracking_file, 'w') as f:
            json.dump(tracking_data, f, indent=2)
        
        print(f"✓ Submission tracking saved: {tracking_file}")
        
        return tracking_data
    
    def create_submission_instructions(self):
        """Create clear instructions for competition submission"""
        print("\n" + "="*60)
        print("COMPETITION SUBMISSION INSTRUCTIONS")
        print("="*60)
        
        instructions = f"""
FAST BERT FEATURES - COMPETITION SUBMISSION READY!

SUBMISSION FILE: {self.competition_file}
PREDICTIONS: {len(pd.read_csv(self.competition_file))} test articles
APPROACH: BERT feature extraction + Logistic Regression
LOADING TIME: ~5-10 seconds (much faster than full fine-tuning)

SUBMISSION STEPS:
1. Go to the competition submission page
2. Upload file: {self.competition_file}
3. Submit and wait for score calculation
4. Compare with Phase 11 baseline (0.61618)

EXPECTED OUTCOMES:
- If score > 0.61618: BERT approach is superior
- If score < 0.61618: Phase 11 baseline is better
- This will determine our next optimization strategy

KEY DIFFERENCES FROM PHASE 11:
- Phase 11: 68.3% Real, 31.7% Fake (heavily biased)
- Fast BERT: 52.3% Real, 47.7% Fake (balanced)
- 59.9% prediction agreement between approaches

IMPORTANT: This submission will reveal whether:
   - BERT approaches provide better generalization
   - Phase 11 bias was actually correct for the competition
   - We need to adjust our approach

COMPETITION TRACKING:
- Current Phase 11: 0.61618 score, Position 619/779
- Fast BERT: Unknown (submitting now)
- Lightning Fast Text: Backup option if needed

NEXT STEPS AFTER SUBMISSION:
1. Monitor competition score
2. Compare with Phase 11 baseline
3. Decide on optimization strategy:
   - If BERT wins: Fine-tune and optimize
   - If Phase 11 wins: Investigate bias and improve
   - If close: Create hybrid ensemble

SUBMISSION STATUS: READY TO SUBMIT!
"""
        
        print(instructions)
        
        # Save instructions to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        instructions_file = f"{self.backup_dir}/fast_bert_submission_instructions_{timestamp}.txt"
        with open(instructions_file, 'w') as f:
            f.write(instructions)
        
        print(f"✓ Instructions saved: {instructions_file}")
    
    def run_immediate_submission(self):
        """Run the complete immediate submission process"""
        print("="*80)
        print("IMMEDIATE COMPETITION SUBMISSION: FAST BERT FEATURES")
        print("="*80)
        
        # Step 1: Validate submission
        if not self.validate_submission():
            print("✗ Submission validation failed. Cannot proceed.")
            return False
        
        # Step 2: Prepare competition submission
        competition_df = self.prepare_competition_submission()
        if competition_df is None:
            print("✗ Failed to prepare competition submission.")
            return False
        
        # Step 3: Generate summary
        tracking_data = self.generate_submission_summary(competition_df)
        
        # Step 4: Create submission instructions
        self.create_submission_instructions()
        
        print("\n" + "="*80)
        print("🎯 FAST BERT FEATURES READY FOR IMMEDIATE COMPETITION SUBMISSION!")
        print("="*80)
        print(f"📁 SUBMISSION FILE: {self.competition_file}")
        print(f"📊 TOTAL PREDICTIONS: {len(competition_df)}")
        print(f"🎯 APPROACH: BERT + Logistic Regression")
        print(f"⚡ LOADING TIME: ~5-10 seconds")
        print(f"📈 BASELINE TO BEAT: Phase 11 (0.61618)")
        
        print("\n🚀 SUBMIT NOW TO TEST COMPETITION PERFORMANCE!")
        print("This will reveal whether BERT approaches can improve our position!")
        
        return True

def main():
    """Main execution function"""
    submitter = FastBERTCompetitionSubmitter()
    success = submitter.run_immediate_submission()
    
    if success:
        print("\n✅ SUBMISSION PREPARATION COMPLETED SUCCESSFULLY!")
        print("🎯 Ready to submit to competition and test performance!")
    else:
        print("\n❌ SUBMISSION PREPARATION FAILED!")
        print("Please check error messages above.")

if __name__ == "__main__":
    main()
