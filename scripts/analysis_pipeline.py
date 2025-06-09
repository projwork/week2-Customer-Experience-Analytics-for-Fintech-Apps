#!/usr/bin/env python3
"""
Analysis Pipeline for Ethiopian Banking App Reviews
Orchestrates sentiment and thematic analysis in a modular workflow
"""

import pandas as pd
import numpy as np
import os
import json
import logging
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import our modular components
from sentiment_analysis import SentimentAnalyzer, calculate_sentiment_aggregations
from thematic_analysis import ThematicAnalyzer, generate_thematic_summary

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('analysis_pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class AnalysisPipeline:
    """
    Comprehensive analysis pipeline for banking app reviews
    """
    
    def __init__(self, use_gpu=False, batch_size=32):
        """
        Initialize the analysis pipeline
        
        Args:
            use_gpu (bool): Whether to use GPU for sentiment analysis
            batch_size (int): Batch size for processing
        """
        self.use_gpu = use_gpu
        self.batch_size = batch_size
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Initialize components
        self.sentiment_analyzer = None
        self.thematic_analyzer = None
        
        # Results storage
        self.results = {
            'metadata': {
                'pipeline_version': '1.0',
                'timestamp': self.timestamp,
                'use_gpu': use_gpu,
                'batch_size': batch_size
            },
            'data_info': {},
            'sentiment_results': {},
            'thematic_results': {},
            'combined_results': {},
            'quality_metrics': {}
        }
    
    def initialize_analyzers(self):
        """Initialize sentiment and thematic analyzers"""
        logger.info("Initializing analysis components...")
        
        try:
            self.sentiment_analyzer = SentimentAnalyzer(use_gpu=self.use_gpu)
            logger.info("✅ Sentiment analyzer initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize sentiment analyzer: {e}")
            return False
        
        try:
            self.thematic_analyzer = ThematicAnalyzer()
            logger.info("✅ Thematic analyzer initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize thematic analyzer: {e}")
            return False
        
        return True
    
    def load_data(self, data_path=None):
        """
        Load and validate input data
        
        Args:
            data_path (str): Path to the data file
            
        Returns:
            pd.DataFrame: Loaded and validated data
        """
        if data_path is None:
            data_path = '../data/processed/cleaned_banking_reviews.csv'
        
        logger.info(f"Loading data from {data_path}...")
        
        try:
            df = pd.read_csv(data_path)
            logger.info(f"✅ Loaded {len(df)} reviews")
            
            # Validate required columns
            required_columns = ['review', 'rating', 'date', 'bank']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                logger.error(f"❌ Missing required columns: {missing_columns}")
                return None
            
            # Store data info
            self.results['data_info'] = {
                'total_reviews': int(len(df)),
                'banks': int(df['bank'].nunique()),
                'bank_distribution': {str(k): int(v) for k, v in df['bank'].value_counts().to_dict().items()},
                'date_range': {
                    'min': str(df['date'].min()),
                    'max': str(df['date'].max())
                },
                'rating_distribution': {str(k): int(v) for k, v in df['rating'].value_counts().sort_index().to_dict().items()}
            }
            
            return df
            
        except FileNotFoundError:
            logger.error(f"❌ Data file not found: {data_path}")
            logger.error("Please run data preprocessing first: python scripts/data_preprocessing.py")
            return None
        except Exception as e:
            logger.error(f"❌ Error loading data: {e}")
            return None
    
    def run_sentiment_analysis(self, df, text_column='review'):
        """
        Run comprehensive sentiment analysis
        
        Args:
            df (pd.DataFrame): Input data
            text_column (str): Column containing review text
            
        Returns:
            pd.DataFrame: Results with sentiment scores
        """
        logger.info("🎭 Starting Sentiment Analysis Phase...")
        
        if self.sentiment_analyzer is None:
            logger.error("Sentiment analyzer not initialized")
            return None
        
        try:
            # Run sentiment analysis
            sentiment_df = self.sentiment_analyzer.analyze_dataframe(
                df, text_column=text_column, batch_size=self.batch_size
            )
            
            # Calculate aggregations
            aggregations = calculate_sentiment_aggregations(sentiment_df)
            
            # Store results (ensure all data is JSON-safe)
            self.results['sentiment_results'] = {
                'total_analyzed': int(len(sentiment_df)),
                'sentiment_coverage': float(aggregations['overall']['sentiment_coverage']),
                'overall_distribution': {
                    'positive': float(aggregations['overall']['positive_ratio']),
                    'negative': float(aggregations['overall']['negative_ratio']),
                    'neutral': float(aggregations['overall']['neutral_ratio'])
                },
                'by_bank': self._convert_to_json_safe(aggregations['by_bank']),
                'by_rating': self._convert_to_json_safe(aggregations['by_rating'])
            }
            
            logger.info(f"✅ Sentiment analysis completed for {len(sentiment_df)} reviews")
            logger.info(f"📊 Sentiment coverage: {aggregations['overall']['sentiment_coverage']:.1f}%")
            
            return sentiment_df
            
        except Exception as e:
            logger.error(f"❌ Sentiment analysis failed: {e}")
            return None
    
    def run_thematic_analysis(self, df, text_column='review'):
        """
        Run comprehensive thematic analysis
        
        Args:
            df (pd.DataFrame): Input data
            text_column (str): Column containing review text
            
        Returns:
            tuple: (thematic_results_dict, theme_assignment_df)
        """
        logger.info("🎯 Starting Thematic Analysis Phase...")
        
        if self.thematic_analyzer is None:
            logger.error("Thematic analyzer not initialized")
            return None, None
        
        try:
            # Run thematic analysis by bank
            thematic_results = self.thematic_analyzer.analyze_themes_by_bank(
                df, text_column=text_column
            )
            
            # Create theme assignment DataFrame
            theme_df = self.thematic_analyzer.create_theme_assignment_dataframe(
                df, thematic_results, text_column=text_column
            )
            
            # Generate summary
            summary = generate_thematic_summary(thematic_results)
            
            # Store results (ensure all data is JSON-safe)
            self.results['thematic_results'] = self._convert_to_json_safe({
                'total_analyzed': int(summary['overall_statistics']['total_reviews_analyzed']),
                'themes_identified': int(summary['overall_statistics']['unique_themes_identified']),
                'themes_per_bank': float(summary['overall_statistics']['themes_per_bank_avg']),
                'bank_top_themes': {
                    str(bank): str(stats['top_theme']) 
                    for bank, stats in summary['bank_comparisons'].items()
                },
                'overall_theme_ranking': {str(k): int(v) for k, v in summary['theme_insights'].items()}
            })
            
            logger.info(f"✅ Thematic analysis completed")
            logger.info(f"🎯 Identified {summary['overall_statistics']['unique_themes_identified']} unique themes")
            
            return thematic_results, theme_df
            
        except Exception as e:
            logger.error(f"❌ Thematic analysis failed: {e}")
            return None, None
    
    def combine_results(self, sentiment_df, theme_df):
        """
        Combine sentiment and thematic analysis results
        
        Args:
            sentiment_df (pd.DataFrame): Sentiment analysis results
            theme_df (pd.DataFrame): Thematic analysis results
            
        Returns:
            pd.DataFrame: Combined analysis results
        """
        logger.info("🔗 Combining sentiment and thematic analysis results...")
        
        try:
            # Merge on review_id
            combined_df = pd.merge(
                sentiment_df, 
                theme_df[['review_id', 'identified_themes', 'primary_theme', 'theme_confidence']], 
                on='review_id', 
                how='left'
            )
            
            # Add derived insights
            combined_df['sentiment_theme_alignment'] = self._calculate_sentiment_theme_alignment(combined_df)
            
            # Calculate cross-analysis metrics
            cross_metrics = self._calculate_cross_analysis_metrics(combined_df)
            
            self.results['combined_results'] = self._convert_to_json_safe(cross_metrics)
            
            logger.info(f"✅ Combined {len(combined_df)} reviews with sentiment and theme data")
            
            return combined_df
            
        except Exception as e:
            logger.error(f"❌ Failed to combine results: {e}")
            return None
    
    def _calculate_sentiment_theme_alignment(self, combined_df):
        """
        Calculate alignment between sentiment and themes
        
        Args:
            combined_df (pd.DataFrame): Combined results
            
        Returns:
            pd.Series: Alignment scores
        """
        alignment_scores = []
        
        for _, row in combined_df.iterrows():
            sentiment = row['ensemble_label']
            primary_theme = row.get('primary_theme', '')
            
            # Define alignment rules
            alignment = 0.5  # Default neutral alignment
            
            if sentiment == 'POSITIVE':
                if any(theme in primary_theme for theme in ['User Interface', 'Feature']):
                    alignment = 0.8
                elif 'Technical Issues' in primary_theme:
                    alignment = 0.2  # Misalignment
            elif sentiment == 'NEGATIVE':
                if any(theme in primary_theme for theme in ['Technical Issues', 'Account Access']):
                    alignment = 0.8
                elif any(theme in primary_theme for theme in ['User Interface', 'Feature']):
                    alignment = 0.2  # Misalignment
            
            alignment_scores.append(alignment)
        
        return pd.Series(alignment_scores)
    
    def _calculate_cross_analysis_metrics(self, combined_df):
        """
        Calculate metrics combining sentiment and thematic analysis
        
        Args:
            combined_df (pd.DataFrame): Combined results
            
        Returns:
            dict: Cross-analysis metrics
        """
        def make_json_safe(obj):
            """Convert any numpy/pandas types to JSON-safe types"""
            if hasattr(obj, 'dtype'):  # numpy types
                if 'int' in str(obj.dtype):
                    return int(obj)
                elif 'float' in str(obj.dtype):
                    return float(obj)
                else:
                    return str(obj)
            elif isinstance(obj, tuple):
                return str(obj)  # Convert tuples to strings
            else:
                return obj
        
        metrics = {}
        
        # Sentiment distribution by theme
        sentiment_by_theme = combined_df.groupby('primary_theme')['ensemble_label'].value_counts(normalize=True)
        metrics['sentiment_by_theme'] = {
            make_json_safe(k): make_json_safe(v) 
            for k, v in sentiment_by_theme.to_dict().items()
        }
        
        # Theme distribution by sentiment
        theme_by_sentiment = combined_df.groupby('ensemble_label')['primary_theme'].value_counts(normalize=True)
        metrics['theme_by_sentiment'] = {
            make_json_safe(k): make_json_safe(v) 
            for k, v in theme_by_sentiment.to_dict().items()
        }
        
        # Alignment metrics
        metrics['average_alignment'] = float(combined_df['sentiment_theme_alignment'].mean())
        metrics['high_alignment_percentage'] = float((combined_df['sentiment_theme_alignment'] > 0.7).mean() * 100)
        
        # Bank-specific insights
        bank_insights = {}
        for bank in combined_df['bank'].unique():
            bank_data = combined_df[combined_df['bank'] == bank]
            bank_insights[str(bank)] = {
                'dominant_sentiment': str(bank_data['ensemble_label'].value_counts().index[0]),
                'dominant_theme': str(bank_data['primary_theme'].value_counts().index[0]),
                'avg_alignment': float(bank_data['sentiment_theme_alignment'].mean()),
                'total_reviews': int(len(bank_data))
            }
        
        metrics['bank_insights'] = bank_insights
        
        return metrics
    
    def _convert_to_json_safe(self, obj):
        """
        Recursively convert all numpy/pandas types to JSON-safe types
        
        Args:
            obj: Object to convert (dict, list, or primitive)
            
        Returns:
            JSON-safe version of the object
        """
        if hasattr(obj, 'dtype'):  # numpy types
            if 'int' in str(obj.dtype):
                return int(obj)
            elif 'float' in str(obj.dtype):
                return float(obj)
            else:
                return str(obj)
        elif isinstance(obj, dict):
            return {
                str(k): self._convert_to_json_safe(v) 
                for k, v in obj.items()
            }
        elif isinstance(obj, (list, tuple)):
            return [self._convert_to_json_safe(item) for item in obj]
        elif isinstance(obj, (int, float, str, bool)) or obj is None:
            return obj
        else:
            return str(obj)  # Convert any other types to string
    
    def calculate_quality_metrics(self, sentiment_df, theme_df, combined_df):
        """
        Calculate quality metrics for the analysis
        
        Args:
            sentiment_df (pd.DataFrame): Sentiment results
            theme_df (pd.DataFrame): Theme results  
            combined_df (pd.DataFrame): Combined results
        """
        logger.info("📊 Calculating quality metrics...")
        
        # Sentiment quality metrics
        sentiment_coverage = len(sentiment_df[sentiment_df['ensemble_label'] != 'NEUTRAL']) / len(sentiment_df) * 100
        sentiment_confidence = sentiment_df['ensemble_confidence'].mean()
        
        # Thematic quality metrics
        theme_coverage = len(theme_df[theme_df['primary_theme'] != 'Unclassified']) / len(theme_df) * 100
        avg_themes_per_review = theme_df['theme_count'].mean()
        
        # Combined quality metrics
        successful_combinations = len(combined_df.dropna(subset=['primary_theme', 'ensemble_label']))
        combination_success_rate = successful_combinations / len(combined_df) * 100
        
        self.results['quality_metrics'] = {
            'sentiment_coverage_pct': float(sentiment_coverage),
            'sentiment_avg_confidence': float(sentiment_confidence),
            'theme_coverage_pct': float(theme_coverage),
            'avg_themes_per_review': float(avg_themes_per_review),
            'combination_success_rate_pct': float(combination_success_rate),
            'total_successful_analysis': int(successful_combinations)
        }
        
        logger.info(f"📈 Quality Metrics:")
        logger.info(f"   • Sentiment Coverage: {sentiment_coverage:.1f}%")
        logger.info(f"   • Theme Coverage: {theme_coverage:.1f}%")
        logger.info(f"   • Combination Success: {combination_success_rate:.1f}%")
    
    def save_results(self, sentiment_df, theme_df, combined_df, output_dir='../data/processed'):
        """
        Save all analysis results
        
        Args:
            sentiment_df (pd.DataFrame): Sentiment results
            theme_df (pd.DataFrame): Theme results
            combined_df (pd.DataFrame): Combined results
            output_dir (str): Output directory
        """
        logger.info("💾 Saving analysis results...")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Save individual results
        sentiment_path = f"{output_dir}/sentiment_analysis_{self.timestamp}.csv"
        theme_path = f"{output_dir}/thematic_analysis_{self.timestamp}.csv"
        combined_path = f"{output_dir}/combined_analysis_{self.timestamp}.csv"
        
        sentiment_df.to_csv(sentiment_path, index=False)
        theme_df.to_csv(theme_path, index=False)
        combined_df.to_csv(combined_path, index=False)
        
        # Save pipeline results as JSON (with comprehensive type conversion)
        results_path = f"{output_dir}/analysis_pipeline_results_{self.timestamp}.json"
        json_safe_results = self._convert_to_json_safe(self.results)
        with open(results_path, 'w') as f:
            json.dump(json_safe_results, f, indent=2)
        
        logger.info(f"✅ Results saved:")
        logger.info(f"   • Sentiment: {sentiment_path}")
        logger.info(f"   • Thematic: {theme_path}")
        logger.info(f"   • Combined: {combined_path}")
        logger.info(f"   • Pipeline Summary: {results_path}")
        
        return {
            'sentiment': sentiment_path,
            'thematic': theme_path,
            'combined': combined_path,
            'summary': results_path
        }
    
    def generate_final_report(self):
        """Generate a comprehensive final report"""
        logger.info("📋 Generating final analysis report...")
        
        print("\n" + "="*80)
        print("🏦 ETHIOPIAN BANKING APPS - CUSTOMER EXPERIENCE ANALYSIS REPORT")
        print("="*80)
        
        # Data overview
        data_info = self.results['data_info']
        print(f"\n📊 DATA OVERVIEW:")
        print(f"   • Total Reviews Analyzed: {data_info['total_reviews']:,}")
        print(f"   • Banks Covered: {data_info['banks']}")
        print(f"   • Date Range: {data_info['date_range']['min']} to {data_info['date_range']['max']}")
        
        print(f"\n🏦 REVIEWS BY BANK:")
        for bank, count in data_info['bank_distribution'].items():
            print(f"   • {bank}: {count:,} reviews")
        
        # Sentiment analysis results
        sentiment_results = self.results['sentiment_results']
        print(f"\n🎭 SENTIMENT ANALYSIS RESULTS:")
        print(f"   • Coverage: {sentiment_results['sentiment_coverage']:.1f}%")
        print(f"   • Positive: {sentiment_results['overall_distribution']['positive']:.1f}%")
        print(f"   • Negative: {sentiment_results['overall_distribution']['negative']:.1f}%")
        print(f"   • Neutral: {sentiment_results['overall_distribution']['neutral']:.1f}%")
        
        # Thematic analysis results
        thematic_results = self.results['thematic_results']
        print(f"\n🎯 THEMATIC ANALYSIS RESULTS:")
        print(f"   • Themes Identified: {thematic_results['themes_identified']}")
        print(f"   • Average Themes per Bank: {thematic_results['themes_per_bank']:.1f}")
        
        print(f"\n🏆 TOP THEMES BY BANK:")
        for bank, theme in thematic_results['bank_top_themes'].items():
            print(f"   • {bank}: {theme}")
        
        # Quality metrics
        quality = self.results['quality_metrics']
        print(f"\n📈 QUALITY METRICS:")
        print(f"   • Sentiment Coverage: {quality['sentiment_coverage_pct']:.1f}%")
        print(f"   • Theme Coverage: {quality['theme_coverage_pct']:.1f}%")
        print(f"   • Analysis Success Rate: {quality['combination_success_rate_pct']:.1f}%")
        
        # KPI Achievement
        print(f"\n✅ KPI ACHIEVEMENT:")
        sentiment_target = quality['sentiment_coverage_pct'] >= 90
        theme_target = thematic_results['themes_identified'] >= 6  # 2+ per bank minimum
        
        print(f"   • Sentiment scores for 90%+ reviews: {'✅' if sentiment_target else '❌'} ({quality['sentiment_coverage_pct']:.1f}%)")
        print(f"   • 3+ themes per bank identified: {'✅' if theme_target else '❌'} ({thematic_results['themes_identified']} total themes)")
        print(f"   • Modular pipeline implemented: ✅")
        
        print(f"\n🎉 ANALYSIS PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*80)
    
    def run_full_pipeline(self, data_path=None, save_results=True):
        """
        Run the complete analysis pipeline
        
        Args:
            data_path (str): Path to input data
            save_results (bool): Whether to save results
            
        Returns:
            dict: Pipeline results and file paths
        """
        logger.info("🚀 Starting Complete Analysis Pipeline...")
        
        # Initialize
        if not self.initialize_analyzers():
            return None
        
        # Load data
        df = self.load_data(data_path)
        if df is None:
            return None
        
        # Run sentiment analysis
        sentiment_df = self.run_sentiment_analysis(df)
        if sentiment_df is None:
            return None
        
        # Run thematic analysis
        thematic_results, theme_df = self.run_thematic_analysis(df)
        if theme_df is None:
            return None
        
        # Combine results
        combined_df = self.combine_results(sentiment_df, theme_df)
        if combined_df is None:
            return None
        
        # Calculate quality metrics
        self.calculate_quality_metrics(sentiment_df, theme_df, combined_df)
        
        # Save results if requested
        file_paths = {}
        if save_results:
            file_paths = self.save_results(sentiment_df, theme_df, combined_df)
        
        # Generate final report
        self.generate_final_report()
        
        return {
            'sentiment_df': sentiment_df,
            'theme_df': theme_df,
            'combined_df': combined_df,
            'pipeline_results': self.results,
            'file_paths': file_paths
        }

def main():
    """Main function to run the analysis pipeline"""
    # Create pipeline instance
    pipeline = AnalysisPipeline(use_gpu=False, batch_size=16)
    
    # Run complete pipeline
    results = pipeline.run_full_pipeline()
    
    if results:
        logger.info("🎉 Pipeline completed successfully!")
        return results
    else:
        logger.error("❌ Pipeline failed!")
        return None

if __name__ == "__main__":
    main() 