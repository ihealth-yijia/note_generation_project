import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set font and style for plots (supports Chinese characters)
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")
plt.style.use('default')

class EvaluationAnalyzer:
    def __init__(self, df):
        """Initialize the analyzer"""
        self.df = df.copy()
        self.scores_df = None
        self.score_metrics = [
            'up_to_date', 'accurate', 'thorough', 'useful',
            'organized', 'comprehensible', 'succinct',
            'synthesized', 'internally_consistent'
        ]
        self.prepare_scores_data()

    def prepare_scores_data(self):
        """Extract and prepare score data"""
        print("📊 Preparing evaluation scores data...")

        # Check if evaluation columns exist
        eval_columns = [col for col in self.df.columns if 'evaluation_claude' in col]
        if not eval_columns:
            raise ValueError("No evaluation columns found! Make sure you have run the evaluation first.")

        # Use the first evaluation column found
        eval_column = eval_columns[0]
        print(f"📋 Using evaluation column: {eval_column}")

        scores_data = []

        for idx, row in self.df.iterrows():
            eval_result = row[eval_column]

            # Skip failed evaluations
            if not isinstance(eval_result, dict) or eval_result.get('processing_status') != 'completed':
                continue

            # Extract basic info
            record = {
                'transcript_id': eval_result.get('transcript_id', row.get('transcript_id', f'row_{idx}')),
                'processing_status': eval_result.get('processing_status', 'unknown'),
                'total_score': eval_result.get('total_score', 0)
            }

            # Extract scores for each metric
            scores = eval_result.get('scores', {})
            for metric in self.score_metrics:
                if metric in scores and isinstance(scores[metric], dict):
                    record[f'{metric}_score'] = scores[metric].get('score', 0)
                    record[f'{metric}_reason'] = scores[metric].get('reason', '')
