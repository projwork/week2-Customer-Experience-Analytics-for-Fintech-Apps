#!/usr/bin/env python3
"""
Data Insertion Script for Bank Reviews PostgreSQL Database

This script loads the cleaned and processed review data from CSV files
into the PostgreSQL database.
"""

import pandas as pd
import psycopg2
import psycopg2.extras
import logging
from datetime import datetime
import os
from typing import Dict, List
import numpy as np

# Database Configuration
DB_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'database': 'bank_reviews',
    'user': 'postgres',
    'password': 'password'  # Change this to your PostgreSQL password
}

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_insertion.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class DataInsertion:
    """Data insertion manager for bank reviews"""
    
    def __init__(self, config: dict = DB_CONFIG):
        self.config = config
        self.connection = None
        self.cursor = None
        self.bank_mapping = {}
    
    def connect(self) -> bool:
        """Establish database connection"""
        try:
            self.connection = psycopg2.connect(**self.config)
            self.cursor = self.connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            logger.info("✅ Database connection established")
            return True
        except psycopg2.Error as e:
            logger.error(f"❌ Database connection failed: {e}")
            return False
    
    def disconnect(self):
        """Close database connection"""
        if self.cursor:
            self.cursor.close()
        if self.connection:
            self.connection.close()
        logger.info("📊 Database connection closed")
    
    def load_bank_mapping(self) -> bool:
        """Load bank name to ID mapping"""
        try:
            self.cursor.execute("SELECT bank_id, bank_name, bank_code FROM banks")
            banks = self.cursor.fetchall()
            
            for bank in banks:
                # Map both bank_name and bank_code to bank_id
                self.bank_mapping[bank['bank_name']] = bank['bank_id']
                self.bank_mapping[bank['bank_code']] = bank['bank_id']
            
            logger.info(f"✅ Loaded {len(banks)} banks: {list(self.bank_mapping.keys())}")
            return True
            
        except psycopg2.Error as e:
            logger.error(f"❌ Failed to load bank mapping: {e}")
            return False
    
    def process_cleaned_data(self, file_path: str) -> pd.DataFrame:
        """Process and validate cleaned data"""
        try:
            logger.info(f"📂 Loading cleaned data from: {file_path}")
            df = pd.read_csv(file_path)
            
            original_count = len(df)
            logger.info(f"📊 Original records: {original_count}")
            
            # Data validation and cleaning
            # 1. Remove rows with missing essential data
            df = df.dropna(subset=['review', 'rating', 'date', 'bank'])
            logger.info(f"📊 After removing missing data: {len(df)}")
            
            # 2. Validate ratings
            df = df[df['rating'].between(1, 5)]
            logger.info(f"📊 After rating validation: {len(df)}")
            
            # 3. Convert date column
            df['date'] = pd.to_datetime(df['date']).dt.date
            
            # 4. Map bank names to IDs
            df['bank_id'] = df['bank'].map(self.bank_mapping)
            df = df.dropna(subset=['bank_id'])
            df['bank_id'] = df['bank_id'].astype(int)
            logger.info(f"📊 After bank mapping: {len(df)}")
            
            # 5. Handle missing processed fields
            df['review_length'] = df['review_length'].fillna(df['review'].str.len())
            df['word_count'] = df['word_count'].fillna(df['review'].str.split().str.len())
            
            # 6. Ensure rating_category is properly set
            if 'rating_category' not in df.columns:
                df['rating_category'] = df['rating'].apply(
                    lambda x: 'Positive' if x >= 4 else ('Negative' if x <= 2 else 'Neutral')
                )
            
            # 7. Extract date components if missing
            df['year'] = df['year'].fillna(pd.to_datetime(df['date']).dt.year)
            df['month'] = df['month'].fillna(pd.to_datetime(df['date']).dt.month)
            df['quarter'] = df['quarter'].fillna(pd.to_datetime(df['date']).dt.quarter)
            
            # 8. Set defaults for boolean fields
            df['rating_valid'] = df['rating_valid'].fillna(True)
            
            logger.info(f"✅ Data processed successfully. Final records: {len(df)}")
            return df
            
        except Exception as e:
            logger.error(f"❌ Error processing cleaned data: {e}")
            return pd.DataFrame()
    
    def process_raw_data(self, file_path: str) -> pd.DataFrame:
        """Process raw scraped data"""
        try:
            logger.info(f"📂 Loading raw data from: {file_path}")
            df = pd.read_csv(file_path)
            
            # Rename columns to match our schema
            column_mapping = {
                'review_text': 'review',
                'bank_name': 'bank'
            }
            df = df.rename(columns=column_mapping)
            
            # Extract bank code from full name
            bank_name_mapping = {
                'Commercial Bank of Ethiopia': 'CBE',
                'Bank of Abyssinia': 'BOA',
                'Dashen Bank': 'Dashen Bank'
            }
            df['bank'] = df['bank'].map(bank_name_mapping)
            
            # Process the data similar to cleaned data
            df = self.process_cleaned_data_logic(df)
            
            logger.info(f"✅ Raw data processed successfully. Records: {len(df)}")
            return df
            
        except Exception as e:
            logger.error(f"❌ Error processing raw data: {e}")
            return pd.DataFrame()
    
    def process_cleaned_data_logic(self, df: pd.DataFrame) -> pd.DataFrame:
        """Common data processing logic"""
        # Remove duplicates based on review text and bank
        original_count = len(df)
        df = df.drop_duplicates(subset=['review', 'bank'])
        logger.info(f"📊 Removed {original_count - len(df)} duplicates")
        
        # Convert date column
        df['date'] = pd.to_datetime(df['date']).dt.date
        
        # Add processed fields
        df['review_length'] = df['review'].str.len()
        df['word_count'] = df['review'].str.split().str.len()
        df['rating_category'] = df['rating'].apply(
            lambda x: 'Positive' if x >= 4 else ('Negative' if x <= 2 else 'Neutral')
        )
        
        # Extract date components
        date_series = pd.to_datetime(df['date'])
        df['year'] = date_series.dt.year
        df['month'] = date_series.dt.month
        df['quarter'] = date_series.dt.quarter
        df['rating_valid'] = True
        
        # Map bank to bank_id
        df['bank_id'] = df['bank'].map(self.bank_mapping)
        df = df.dropna(subset=['bank_id'])
        df['bank_id'] = df['bank_id'].astype(int)
        
        return df
    
    def insert_reviews_batch(self, df: pd.DataFrame, batch_size: int = 1000) -> bool:
        """Insert reviews in batches for better performance"""
        try:
            total_records = len(df)
            logger.info(f"📊 Inserting {total_records} reviews in batches of {batch_size}")
            
            # Prepare insert query
            insert_sql = """
            INSERT INTO reviews (
                bank_id, review_text, rating, review_date, source,
                review_length, word_count, rating_category,
                year, month, quarter, rating_valid
            ) VALUES %s
            ON CONFLICT DO NOTHING
            """
            
            successful_inserts = 0
            
            # Process in batches
            for i in range(0, total_records, batch_size):
                batch_df = df.iloc[i:i+batch_size]
                
                # Prepare batch data
                batch_data = []
                for _, row in batch_df.iterrows():
                    batch_data.append((
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
                
                # Insert batch
                psycopg2.extras.execute_values(
                    self.cursor, insert_sql, batch_data,
                    template=None, page_size=batch_size
                )
                
                successful_inserts += len(batch_data)
                
                # Progress logging
                progress = (i + batch_size) / total_records * 100
                logger.info(f"📈 Progress: {min(progress, 100):.1f}% ({successful_inserts}/{total_records})")
            
            self.connection.commit()
            logger.info(f"✅ Successfully inserted {successful_inserts} reviews")
            return True
            
        except psycopg2.Error as e:
            logger.error(f"❌ Failed to insert reviews: {e}")
            self.connection.rollback()
            return False
    
    def get_statistics(self) -> Dict:
        """Get database statistics after insertion"""
        try:
            stats = {}
            
            # Total reviews
            self.cursor.execute("SELECT COUNT(*) as count FROM reviews")
            stats['total_reviews'] = self.cursor.fetchone()['count']
            
            # Reviews by bank
            self.cursor.execute("""
                SELECT b.bank_name, COUNT(r.review_id) as review_count
                FROM banks b
                LEFT JOIN reviews r ON b.bank_id = r.bank_id
                GROUP BY b.bank_name
                ORDER BY review_count DESC
            """)
            stats['reviews_by_bank'] = dict(self.cursor.fetchall())
            
            # Rating distribution
            self.cursor.execute("""
                SELECT rating, COUNT(*) as count
                FROM reviews
                GROUP BY rating
                ORDER BY rating
            """)
            stats['rating_distribution'] = dict(self.cursor.fetchall())
            
            # Date range
            self.cursor.execute("""
                SELECT MIN(review_date) as earliest, MAX(review_date) as latest
                FROM reviews
            """)
            date_range = self.cursor.fetchone()
            stats['date_range'] = {
                'earliest': date_range['earliest'],
                'latest': date_range['latest']
            }
            
            return stats
            
        except psycopg2.Error as e:
            logger.error(f"❌ Error getting statistics: {e}")
            return {}

def load_all_data() -> bool:
    """Main function to load all available data"""
    logger.info("🚀 Starting data insertion process")
    
    inserter = DataInsertion()
    
    try:
        # Connect to database
        if not inserter.connect():
            return False
        
        # Load bank mapping
        if not inserter.load_bank_mapping():
            return False
        
        success_count = 0
        total_records = 0
        
        # 1. Load cleaned data if available
        cleaned_file = '../data/processed/cleaned_banking_reviews.csv'
        if os.path.exists(cleaned_file):
            logger.info("📂 Loading cleaned data...")
            df_cleaned = inserter.process_cleaned_data(cleaned_file)
            if not df_cleaned.empty:
                if inserter.insert_reviews_batch(df_cleaned):
                    success_count += 1
                    total_records += len(df_cleaned)
        
        # 2. Load raw scraped data
        data_folder = '../data'
        if os.path.exists(data_folder):
            for filename in os.listdir(data_folder):
                if filename.endswith('_reviews_') and filename.endswith('.csv'):
                    file_path = os.path.join(data_folder, filename)
                    logger.info(f"📂 Loading raw data: {filename}")
                    
                    df_raw = inserter.process_raw_data(file_path)
                    if not df_raw.empty:
                        if inserter.insert_reviews_batch(df_raw):
                            success_count += 1
                            total_records += len(df_raw)
        
        # Get final statistics
        stats = inserter.get_statistics()
        
        logger.info("🎉 Data insertion completed!")
        logger.info(f"📊 Final Statistics:")
        logger.info(f"   • Total Reviews: {stats.get('total_reviews', 0):,}")
        logger.info(f"   • Reviews by Bank: {stats.get('reviews_by_bank', {})}")
        logger.info(f"   • Rating Distribution: {stats.get('rating_distribution', {})}")
        logger.info(f"   • Date Range: {stats.get('date_range', {})}")
        
        # Check if we meet the KPI requirement
        if stats.get('total_reviews', 0) >= 1000:
            logger.info("✅ KPI MET: Database contains >1,000 entries")
        else:
            logger.warning(f"⚠️ KPI NOT MET: Database contains only {stats.get('total_reviews', 0)} entries")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Data insertion process failed: {e}")
        return False
    
    finally:
        inserter.disconnect()

if __name__ == "__main__":
    print("=" * 60)
    print("Bank Reviews Data Insertion")
    print("=" * 60)
    
    success = load_all_data()
    
    if success:
        print("\n✅ SUCCESS: Data insertion completed successfully!")
        print("📚 Next steps:")
        print("   1. Connect to your database using the provided connection string")
        print("   2. Query the 'reviews' and 'banks' tables")
        print("   3. Use the data for analysis and reporting")
    else:
        print("\n❌ FAILED: Data insertion encountered errors!")
        print("📚 Please check the logs and fix any issues") 