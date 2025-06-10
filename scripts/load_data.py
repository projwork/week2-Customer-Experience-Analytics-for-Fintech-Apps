#!/usr/bin/env python3
"""
Load Review Data into PostgreSQL Database
"""

import pandas as pd
import psycopg2
import psycopg2.extras
import logging
import os
from datetime import datetime

# Configuration
DB_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'database': 'bank_reviews',
    'user': 'postgres',
    'password': 'BkPassw0rd'  # CHANGE THIS!
}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_bank_mapping():
    """Get bank name to ID mapping"""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        
        cursor.execute("SELECT bank_id, bank_name, bank_code FROM banks")
        banks = cursor.fetchall()
        
        mapping = {}
        for bank in banks:
            mapping[bank['bank_name']] = bank['bank_id']
            mapping[bank['bank_code']] = bank['bank_id']
        
        # Add variations
        mapping['Commercial Bank of Ethiopia'] = mapping.get('CBE')
        mapping['Bank of Abyssinia'] = mapping.get('BOA')
        
        cursor.close()
        conn.close()
        return mapping
        
    except Exception as e:
        logger.error(f"Error getting bank mapping: {e}")
        return {}

def process_data(file_path, bank_mapping):
    """Process CSV data for insertion"""
    try:
        logger.info(f"Processing file: {file_path}")
        df = pd.read_csv(file_path)
        
        # Check column names and rename if needed
        if 'review_text' in df.columns:
            df = df.rename(columns={'review_text': 'review'})
        if 'bank_name' in df.columns:
            df = df.rename(columns={'bank_name': 'bank'})
        
        # Map bank names to simplified versions
        bank_name_map = {
            'Commercial Bank of Ethiopia': 'CBE',
            'Bank of Abyssinia': 'BOA',
            'Dashen Bank': 'Dashen Bank'
        }
        df['bank'] = df['bank'].map(bank_name_map).fillna(df['bank'])
        
        # Remove missing data
        df = df.dropna(subset=['review', 'rating', 'date', 'bank'])
        
        # Validate ratings
        df = df[df['rating'].between(1, 5)]
        
        # Convert date
        df['date'] = pd.to_datetime(df['date']).dt.date
        
        # Add derived fields if missing
        if 'review_length' not in df.columns:
            df['review_length'] = df['review'].str.len()
        if 'word_count' not in df.columns:
            df['word_count'] = df['review'].str.split().str.len()
        if 'rating_category' not in df.columns:
            df['rating_category'] = df['rating'].apply(
                lambda x: 'Positive' if x >= 4 else ('Negative' if x <= 2 else 'Neutral')
            )
        
        # Extract date components
        date_series = pd.to_datetime(df['date'])
        if 'year' not in df.columns:
            df['year'] = date_series.dt.year
        if 'month' not in df.columns:
            df['month'] = date_series.dt.month
        if 'quarter' not in df.columns:
            df['quarter'] = date_series.dt.quarter
        if 'rating_valid' not in df.columns:
            df['rating_valid'] = True
        
        # Map bank to bank_id
        df['bank_id'] = df['bank'].map(bank_mapping)
        df = df.dropna(subset=['bank_id'])
        df['bank_id'] = df['bank_id'].astype(int)
        
        logger.info(f"Processed {len(df)} records from {file_path}")
        return df
        
    except Exception as e:
        logger.error(f"Error processing data from {file_path}: {e}")
        return pd.DataFrame()

def insert_data(df):
    """Insert data into PostgreSQL"""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()
        
        insert_sql = """
        INSERT INTO reviews (
            bank_id, review_text, rating, review_date, source,
            review_length, word_count, rating_category,
            year, month, quarter, rating_valid
        ) VALUES %s
        ON CONFLICT DO NOTHING
        """
        
        # Prepare data for insertion
        data_tuples = []
        for _, row in df.iterrows():
            data_tuples.append((
                int(row['bank_id']),
                str(row['review']),
                int(row['rating']),
                row['date'],
                str(row.get('source', 'Google Play')),
                int(row['review_length']),
                int(row['word_count']),
                str(row['rating_category']),
                int(row['year']),
                int(row['month']),
                int(row['quarter']),
                bool(row['rating_valid'])
            ))
        
        # Batch insert
        psycopg2.extras.execute_values(
            cursor, insert_sql, data_tuples, page_size=1000
        )
        
        conn.commit()
        logger.info(f"Successfully inserted {len(data_tuples)} records")
        
        cursor.close()
        conn.close()
        return True
        
    except Exception as e:
        logger.error(f"Error inserting data: {e}")
        return False

def get_stats():
    """Get database statistics"""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        
        # Total reviews
        cursor.execute("SELECT COUNT(*) as count FROM reviews")
        total = cursor.fetchone()['count']
        
        # By bank
        cursor.execute("""
            SELECT b.bank_name, COUNT(r.review_id) as count
            FROM banks b
            LEFT JOIN reviews r ON b.bank_id = r.bank_id
            GROUP BY b.bank_name
        """)
        by_bank = dict(cursor.fetchall())
        
        cursor.close()
        conn.close()
        
        return {'total': total, 'by_bank': by_bank}
        
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        return {}

def main():
    print("Loading data into PostgreSQL database...")
    
    # Get bank mapping
    bank_mapping = get_bank_mapping()
    if not bank_mapping:
        print("❌ Failed to get bank mapping")
        return
    
    # Load cleaned data
    cleaned_file = '../data/processed/cleaned_banking_reviews.csv'
    if os.path.exists(cleaned_file):
        df = process_data(cleaned_file, bank_mapping)
        if not df.empty:
            insert_data(df)
    
    # Load raw data files
    data_dir = '../data'
    if os.path.exists(data_dir):
        for filename in os.listdir(data_dir):
            if filename.endswith('_reviews_') and filename.endswith('.csv'):
                file_path = os.path.join(data_dir, filename)
                df = process_data(file_path, bank_mapping)
                if not df.empty:
                    insert_data(df)
    
    # Show final statistics
    stats = get_stats()
    print(f"\n📊 Final Statistics:")
    print(f"   Total Reviews: {stats.get('total', 0):,}")
    print(f"   By Bank: {stats.get('by_bank', {})}")
    
    if stats.get('total', 0) >= 1000:
        print("✅ KPI MET: Database contains >1,000 entries")
    else:
        print(f"⚠️ Only {stats.get('total', 0)} entries loaded")
    
    print("✅ Data loading completed!")

if __name__ == "__main__":
    main() 