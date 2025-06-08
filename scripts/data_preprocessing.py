#!/usr/bin/env python3
"""
Data Preprocessing Script for Ethiopian Banking App Reviews
Cleans, merges, and standardizes review data from multiple banks
"""

import pandas as pd
import numpy as np
import os
import glob
from datetime import datetime
import re
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('preprocessing.log'),
        logging.StreamHandler()
    ]
)

def setup_output_directory():
    """Create output directory if it doesn't exist"""
    if not os.path.exists('data/processed'):
        os.makedirs('data/processed')
        logging.info("Created data/processed directory")

def find_latest_files():
    """Find the most recent CSV files for each bank"""
    banks = ['CBE', 'BOA', 'Dashen_Bank']
    latest_files = {}
    
    for bank in banks:
        pattern = f'data/{bank}_reviews_*.csv'
        files = glob.glob(pattern)
        
        if files:
            # Sort by modification time to get the latest
            latest_file = max(files, key=os.path.getmtime)
            latest_files[bank] = latest_file
            logging.info(f"Found latest file for {bank}: {latest_file}")
        else:
            logging.warning(f"No files found for {bank}")
    
    return latest_files

def load_and_validate_data(file_path, bank_name):
    """Load data and perform initial validation"""
    try:
        df = pd.read_csv(file_path, encoding='utf-8')
        logging.info(f"Loaded {len(df)} reviews for {bank_name}")
        
        # Basic validation
        required_columns = ['review_text', 'rating', 'date', 'bank_name', 'source', 'app_id']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            logging.error(f"Missing columns in {bank_name}: {missing_columns}")
            return None
        
        return df
    
    except Exception as e:
        logging.error(f"Error loading {bank_name} data: {e}")
        return None

def clean_text(text):
    """Clean and standardize text data"""
    if pd.isna(text) or text == '':
        return ''
    
    # Convert to string
    text = str(text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Remove or replace problematic characters (keep essential punctuation)
    text = re.sub(r'[^\w\s\.\!\?\,\;\:\-\(\)\'\"]', ' ', text)
    
    return text

def standardize_bank_names(df):
    """Standardize bank names for consistency"""
    bank_mapping = {
        'Commercial Bank of Ethiopia': 'CBE',
        'Bank of Abyssinia': 'BOA', 
        'Dashen Bank': 'Dashen Bank'
    }
    
    df['bank'] = df['bank_name'].map(bank_mapping)
    
    # Fill any unmapped values
    df['bank'] = df['bank'].fillna(df['bank_name'])
    
    return df

def clean_ratings(df):
    """Clean and validate rating data"""
    # Convert to numeric
    df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
    
    # Filter valid ratings (1-5)
    invalid_ratings = df[(df['rating'] < 1) | (df['rating'] > 5) | df['rating'].isna()]
    if len(invalid_ratings) > 0:
        logging.warning(f"Found {len(invalid_ratings)} invalid ratings, will be handled")
    
    # For analysis, we'll keep all data but flag invalid ratings
    df['rating_valid'] = ((df['rating'] >= 1) & (df['rating'] <= 5))
    
    return df

def standardize_dates(df):
    """Standardize date format to YYYY-MM-DD"""
    try:
        # Convert to datetime
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        
        # Convert to standard format
        df['date'] = df['date'].dt.strftime('%Y-%m-%d')
        
        # Count invalid dates
        invalid_dates = df[df['date'].isna()].shape[0]
        if invalid_dates > 0:
            logging.warning(f"Found {invalid_dates} invalid dates")
        
        return df
    
    except Exception as e:
        logging.error(f"Error standardizing dates: {e}")
        return df

def remove_duplicates(df):
    """Remove duplicate reviews based on content similarity"""
    initial_count = len(df)
    
    # Remove exact duplicates
    df = df.drop_duplicates(subset=['review_text', 'rating', 'bank'], keep='first')
    
    # Remove reviews with identical text (case-insensitive)
    df['review_lower'] = df['review_text'].str.lower()
    df = df.drop_duplicates(subset=['review_lower', 'bank'], keep='first')
    df = df.drop('review_lower', axis=1)
    
    final_count = len(df)
    removed = initial_count - final_count
    
    logging.info(f"Removed {removed} duplicate reviews ({removed/initial_count*100:.1f}%)")
    
    return df

def handle_missing_data(df):
    """Handle missing data according to business rules"""
    initial_count = len(df)
    
    # Remove rows with missing critical data
    critical_columns = ['review_text', 'bank']
    df = df.dropna(subset=critical_columns)
    
    # Fill missing ratings with median by bank
    for bank in df['bank'].unique():
        bank_median = df[df['bank'] == bank]['rating'].median()
        df.loc[(df['bank'] == bank) & (df['rating'].isna()), 'rating'] = bank_median
    
    # Fill missing dates with a placeholder
    df['date'] = df['date'].fillna('1900-01-01')
    
    # Fill missing source
    df['source'] = df['source'].fillna('Google Play')
    
    final_count = len(df)
    removed = initial_count - final_count
    
    logging.info(f"Removed {removed} rows due to missing critical data")
    
    return df

def add_derived_features(df):
    """Add useful derived features for analysis"""
    # Review length
    df['review_length'] = df['review_text'].str.len()
    
    # Review word count
    df['word_count'] = df['review_text'].str.split().str.len()
    
    # Rating categories
    df['rating_category'] = pd.cut(df['rating'], 
                                  bins=[0, 2, 3, 5], 
                                  labels=['Negative', 'Neutral', 'Positive'])
    
    # Date features
    df['date_parsed'] = pd.to_datetime(df['date'], errors='coerce')
    df['year'] = df['date_parsed'].dt.year
    df['month'] = df['date_parsed'].dt.month
    df['quarter'] = df['date_parsed'].dt.quarter
    
    return df

def generate_quality_report(df):
    """Generate data quality report"""
    report = {
        'total_reviews': len(df),
        'banks': df['bank'].nunique(),
        'date_range': {
            'min': df['date'].min(),
            'max': df['date'].max()
        },
        'missing_data_pct': {
            'review_text': (df['review_text'].isna().sum() / len(df)) * 100,
            'rating': (df['rating'].isna().sum() / len(df)) * 100,
            'date': (df['date'].isna().sum() / len(df)) * 100,
        },
        'reviews_per_bank': df['bank'].value_counts().to_dict(),
        'rating_distribution': df['rating'].value_counts().sort_index().to_dict(),
        'avg_review_length': df['review_length'].mean(),
        'avg_word_count': df['word_count'].mean()
    }
    
    return report

def save_processed_data(df, filename='cleaned_banking_reviews.csv'):
    """Save the processed dataset"""
    output_path = f'data/processed/{filename}'
    
    # Select final columns for output
    final_columns = [
        'review_text', 'rating', 'date', 'bank', 'source',
        'review_length', 'word_count', 'rating_category',
        'year', 'month', 'quarter', 'rating_valid'
    ]
    
    # Ensure all columns exist
    available_columns = [col for col in final_columns if col in df.columns]
    
    df_final = df[available_columns].copy()
    
    # Rename for consistency with requirements
    df_final = df_final.rename(columns={
        'review_text': 'review',
        'rating': 'rating',
        'date': 'date',
        'bank': 'bank',
        'source': 'source'
    })
    
    # Save to CSV
    df_final.to_csv(output_path, index=False, encoding='utf-8')
    logging.info(f"Saved processed data to {output_path}")
    
    return output_path, df_final

def main():
    """Main preprocessing pipeline"""
    print("🚀 Starting Ethiopian Banking Reviews Data Preprocessing")
    print("=" * 60)
    
    # Setup
    setup_output_directory()
    
    # Find latest data files
    latest_files = find_latest_files()
    
    if not latest_files:
        logging.error("No data files found!")
        return
    
    # Load and combine all data
    all_dataframes = []
    
    for bank, file_path in latest_files.items():
        df = load_and_validate_data(file_path, bank)
        if df is not None:
            all_dataframes.append(df)
    
    if not all_dataframes:
        logging.error("No valid data loaded!")
        return
    
    # Combine all data
    combined_df = pd.concat(all_dataframes, ignore_index=True)
    logging.info(f"Combined data: {len(combined_df)} total reviews")
    
    # Preprocessing steps
    print("\n📊 Preprocessing Steps:")
    print("-" * 30)
    
    print("1. Cleaning text data...")
    combined_df['review_text'] = combined_df['review_text'].apply(clean_text)
    
    print("2. Standardizing bank names...")
    combined_df = standardize_bank_names(combined_df)
    
    print("3. Cleaning ratings...")
    combined_df = clean_ratings(combined_df)
    
    print("4. Standardizing dates...")
    combined_df = standardize_dates(combined_df)
    
    print("5. Removing duplicates...")
    combined_df = remove_duplicates(combined_df)
    
    print("6. Handling missing data...")
    combined_df = handle_missing_data(combined_df)
    
    print("7. Adding derived features...")
    combined_df = add_derived_features(combined_df)
    
    # Generate quality report
    print("\n📈 Generating Quality Report...")
    quality_report = generate_quality_report(combined_df)
    
    # Save processed data
    print("\n💾 Saving processed data...")
    output_path, final_df = save_processed_data(combined_df)
    
    # Display results
    print("\n" + "=" * 60)
    print("✅ PREPROCESSING COMPLETED!")
    print("=" * 60)
    print(f"📊 Total Reviews: {quality_report['total_reviews']:,}")
    print(f"🏦 Banks: {quality_report['banks']}")
    print(f"📅 Date Range: {quality_report['date_range']['min']} to {quality_report['date_range']['max']}")
    print(f"📝 Avg Review Length: {quality_report['avg_review_length']:.1f} characters")
    print(f"💬 Avg Word Count: {quality_report['avg_word_count']:.1f} words")
    
    print("\n🏦 Reviews per Bank:")
    for bank, count in quality_report['reviews_per_bank'].items():
        print(f"   • {bank}: {count:,} reviews")
    
    print("\n⭐ Rating Distribution:")
    for rating, count in quality_report['rating_distribution'].items():
        print(f"   • {rating} stars: {count:,} reviews")
    
    print(f"\n📁 Processed data saved to: {output_path}")
    
    # Missing data summary
    missing_pct = quality_report['missing_data_pct']
    total_missing = max(missing_pct.values())
    print(f"📊 Missing Data: {total_missing:.2f}% (Target: <5%)")
    
    if total_missing < 5:
        print("✅ Missing data target achieved!")
    else:
        print("⚠️ Missing data above target threshold")
    
    print("\n🎉 Data is ready for analysis!")
    
    return final_df, quality_report

if __name__ == "__main__":
    main() 