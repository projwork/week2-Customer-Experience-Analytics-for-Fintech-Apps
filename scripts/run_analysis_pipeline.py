#!/usr/bin/env python3
"""
Runner Script for Complete Sentiment and Thematic Analysis Pipeline
Simple interface to run the full analysis on Ethiopian banking app reviews
"""

import argparse
import sys
import logging
from analysis_pipeline import AnalysisPipeline

def main():
    parser = argparse.ArgumentParser(
        description="Run sentiment and thematic analysis on banking app reviews"
    )
    
    parser.add_argument(
        '--data-path', 
        type=str, 
        default='../data/processed/cleaned_banking_reviews.csv',
        help='Path to the preprocessed data file'
    )
    
    parser.add_argument(
        '--use-gpu', 
        action='store_true',
        help='Use GPU for DistilBERT sentiment analysis (requires CUDA)'
    )
    
    parser.add_argument(
        '--batch-size', 
        type=int, 
        default=16,
        help='Batch size for processing (adjust based on memory)'
    )
    
    parser.add_argument(
        '--output-dir', 
        type=str, 
        default='../data/processed',
        help='Directory to save analysis results'
    )
    
    parser.add_argument(
        '--sample-size', 
        type=int, 
        default=None,
        help='Number of reviews to analyze (for testing)'
    )
    
    parser.add_argument(
        '--log-level', 
        type=str, 
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level'
    )
    
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    print("🏦 Ethiopian Banking Apps - Sentiment & Thematic Analysis")
    print("="*60)
    print(f"📁 Data Path: {args.data_path}")
    print(f"🖥️ GPU Usage: {args.use_gpu}")
    print(f"📦 Batch Size: {args.batch_size}")
    print(f"💾 Output Directory: {args.output_dir}")
    if args.sample_size:
        print(f"🎯 Sample Size: {args.sample_size} reviews")
    print("="*60)
    
    try:
        # Initialize pipeline
        pipeline = AnalysisPipeline(
            use_gpu=args.use_gpu,
            batch_size=args.batch_size
        )
        
        # Load data
        import pandas as pd
        df = pd.read_csv(args.data_path)
        
        # Apply sample size if specified
        if args.sample_size and args.sample_size < len(df):
            print(f"📊 Using sample of {args.sample_size} reviews for analysis")
            df = df.sample(n=args.sample_size, random_state=42)
        
        print(f"📊 Analyzing {len(df)} reviews...")
        
        # Run pipeline
        results = pipeline.run_full_pipeline(save_results=True)
        
        if results:
            print("\n🎉 Analysis completed successfully!")
            print("\n📁 Generated Files:")
            for file_type, path in results['file_paths'].items():
                print(f"   • {file_type.title()}: {path}")
            
            print("\n✅ KPI Achievement Summary:")
            quality = results['pipeline_results']['quality_metrics']
            print(f"   • Sentiment Coverage: {quality['sentiment_coverage_pct']:.1f}% (Target: 90%+)")
            print(f"   • Theme Coverage: {quality['theme_coverage_pct']:.1f}%")
            print(f"   • Successfully Analyzed: {quality['total_successful_analysis']} reviews")
            
        else:
            print("❌ Pipeline failed. Check logs for details.")
            sys.exit(1)
            
    except FileNotFoundError:
        print(f"❌ Data file not found: {args.data_path}")
        print("Please run data preprocessing first:")
        print("python scripts/data_preprocessing.py")
        sys.exit(1)
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        logging.exception("Pipeline error details:")
        sys.exit(1)

if __name__ == "__main__":
    main() 