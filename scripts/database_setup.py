#!/usr/bin/env python3
"""
PostgreSQL Database Setup for Ethiopian Banking Apps Review Analysis

This script creates the database schema and sets up the initial structure
for storing scraped and processed banking app reviews.

Database: bank_reviews
Tables: banks, reviews
"""

import psycopg2
import psycopg2.extras
import logging
from datetime import datetime
import os
from typing import Optional

# Database Configuration
DB_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'database': 'bank_reviews',
    'user': 'postgres',  # Default PostgreSQL user
    'password': 'password'  # Change this to your PostgreSQL password
}

# Connection string template
CONNECTION_STRING = "postgresql://postgres:Password@localhost:5432/bank_reviews"

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('database_setup.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class BankReviewsDB:
    """Database manager for bank reviews system"""
    
    def __init__(self, config: dict = DB_CONFIG):
        self.config = config
        self.connection = None
        self.cursor = None
    
    def get_connection_string(self) -> str:
        """Generate PostgreSQL connection string"""
        return CONNECTION_STRING.format(**self.config)
    
    def connect(self) -> bool:
        """Establish database connection"""
        try:
            self.connection = psycopg2.connect(**self.config)
            self.cursor = self.connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            logger.info("✅ Database connection established successfully")
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
    
    def create_database(self) -> bool:
        """Create the bank_reviews database if it doesn't exist"""
        try:
            # Connect to PostgreSQL server (not specific database)
            temp_config = self.config.copy()
            temp_config['database'] = 'postgres'  # Default database
            
            temp_conn = psycopg2.connect(**temp_config)
            temp_conn.autocommit = True
            temp_cursor = temp_conn.cursor()
            
            # Check if database exists
            temp_cursor.execute(
                "SELECT 1 FROM pg_database WHERE datname = %s",
                (self.config['database'],)
            )
            
            if not temp_cursor.fetchone():
                temp_cursor.execute(f"CREATE DATABASE {self.config['database']}")
                logger.info(f"✅ Database '{self.config['database']}' created successfully")
            else:
                logger.info(f"📊 Database '{self.config['database']}' already exists")
            
            temp_cursor.close()
            temp_conn.close()
            return True
            
        except psycopg2.Error as e:
            logger.error(f"❌ Failed to create database: {e}")
            return False
    
    def create_tables(self) -> bool:
        """Create the required tables with proper schema"""
        try:
            # Banks table schema
            banks_table_sql = """
            CREATE TABLE IF NOT EXISTS banks (
                bank_id SERIAL PRIMARY KEY,
                bank_name VARCHAR(255) NOT NULL UNIQUE,
                bank_code VARCHAR(10) NOT NULL UNIQUE,
                app_id VARCHAR(255) NOT NULL UNIQUE,
                full_name VARCHAR(255) NOT NULL,
                country VARCHAR(100) DEFAULT 'Ethiopia',
                sector VARCHAR(100) DEFAULT 'Banking',
                app_store_url TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """
            
            # Reviews table schema
            reviews_table_sql = """
            CREATE TABLE IF NOT EXISTS reviews (
                review_id SERIAL PRIMARY KEY,
                bank_id INTEGER NOT NULL REFERENCES banks(bank_id) ON DELETE CASCADE,
                review_text TEXT NOT NULL,
                rating INTEGER NOT NULL CHECK (rating >= 1 AND rating <= 5),
                review_date DATE NOT NULL,
                source VARCHAR(100) DEFAULT 'Google Play',
                
                -- Processed fields from cleaning
                review_length INTEGER,
                word_count INTEGER,
                rating_category VARCHAR(20),
                year INTEGER,
                month INTEGER,
                quarter INTEGER,
                rating_valid BOOLEAN DEFAULT TRUE,
                
                -- Metadata
                app_id VARCHAR(255),
                language VARCHAR(10) DEFAULT 'en',
                is_processed BOOLEAN DEFAULT FALSE,
                is_duplicate BOOLEAN DEFAULT FALSE,
                
                -- Indexing and tracking
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                
                -- Constraints
                CONSTRAINT valid_rating_category CHECK (
                    rating_category IN ('Positive', 'Negative', 'Neutral')
                ),
                CONSTRAINT valid_quarter CHECK (quarter >= 1 AND quarter <= 4),
                CONSTRAINT valid_month CHECK (month >= 1 AND month <= 12)
            );
            """
            
            # Create indexes for better performance
            indexes_sql = [
                "CREATE INDEX IF NOT EXISTS idx_reviews_bank_id ON reviews(bank_id);",
                "CREATE INDEX IF NOT EXISTS idx_reviews_rating ON reviews(rating);",
                "CREATE INDEX IF NOT EXISTS idx_reviews_date ON reviews(review_date);",
                "CREATE INDEX IF NOT EXISTS idx_reviews_category ON reviews(rating_category);",
                "CREATE INDEX IF NOT EXISTS idx_reviews_year ON reviews(year);",
                "CREATE INDEX IF NOT EXISTS idx_reviews_processed ON reviews(is_processed);",
                "CREATE INDEX IF NOT EXISTS idx_banks_code ON banks(bank_code);",
                "CREATE INDEX IF NOT EXISTS idx_banks_app_id ON banks(app_id);"
            ]
            
            # Execute table creation
            self.cursor.execute(banks_table_sql)
            logger.info("✅ Banks table created/verified successfully")
            
            self.cursor.execute(reviews_table_sql)
            logger.info("✅ Reviews table created/verified successfully")
            
            # Create indexes
            for index_sql in indexes_sql:
                self.cursor.execute(index_sql)
            logger.info("✅ Database indexes created/verified successfully")
            
            # Create update timestamp triggers
            trigger_sql = """
            CREATE OR REPLACE FUNCTION update_updated_at_column()
            RETURNS TRIGGER AS $$
            BEGIN
                NEW.updated_at = CURRENT_TIMESTAMP;
                RETURN NEW;
            END;
            $$ language 'plpgsql';
            
            DROP TRIGGER IF EXISTS update_banks_updated_at ON banks;
            CREATE TRIGGER update_banks_updated_at 
                BEFORE UPDATE ON banks 
                FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
                
            DROP TRIGGER IF EXISTS update_reviews_updated_at ON reviews;
            CREATE TRIGGER update_reviews_updated_at 
                BEFORE UPDATE ON reviews 
                FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
            """
            
            self.cursor.execute(trigger_sql)
            logger.info("✅ Update timestamp triggers created successfully")
            
            self.connection.commit()
            return True
            
        except psycopg2.Error as e:
            logger.error(f"❌ Failed to create tables: {e}")
            self.connection.rollback()
            return False
    
    def insert_banks_data(self) -> bool:
        """Insert initial bank data"""
        try:
            banks_data = [
                {
                    'bank_name': 'CBE',
                    'bank_code': 'CBE',
                    'app_id': 'com.combanketh.mobilebanking',
                    'full_name': 'Commercial Bank of Ethiopia',
                    'app_store_url': 'https://play.google.com/store/apps/details?id=com.combanketh.mobilebanking'
                },
                {
                    'bank_name': 'BOA',
                    'bank_code': 'BOA',
                    'app_id': 'com.boa.boaMobileBanking',
                    'full_name': 'Bank of Abyssinia',
                    'app_store_url': 'https://play.google.com/store/apps/details?id=com.boa.boaMobileBanking'
                },
                {
                    'bank_name': 'Dashen Bank',
                    'bank_code': 'DASH',
                    'app_id': 'com.dashen.dashensuperapp',
                    'full_name': 'Dashen Bank',
                    'app_store_url': 'https://play.google.com/store/apps/details?id=com.dashen.dashensuperapp'
                }
            ]
            
            insert_sql = """
            INSERT INTO banks (bank_name, bank_code, app_id, full_name, app_store_url)
            VALUES (%(bank_name)s, %(bank_code)s, %(app_id)s, %(full_name)s, %(app_store_url)s)
            ON CONFLICT (bank_name) DO UPDATE SET
                app_id = EXCLUDED.app_id,
                full_name = EXCLUDED.full_name,
                app_store_url = EXCLUDED.app_store_url,
                updated_at = CURRENT_TIMESTAMP
            """
            
            for bank in banks_data:
                self.cursor.execute(insert_sql, bank)
            
            self.connection.commit()
            logger.info(f"✅ Successfully inserted/updated {len(banks_data)} banks")
            return True
            
        except psycopg2.Error as e:
            logger.error(f"❌ Failed to insert banks data: {e}")
            self.connection.rollback()
            return False
    
    def get_bank_id(self, bank_name: str) -> Optional[int]:
        """Get bank ID by bank name"""
        try:
            self.cursor.execute(
                "SELECT bank_id FROM banks WHERE bank_name = %s OR bank_code = %s",
                (bank_name, bank_name)
            )
            result = self.cursor.fetchone()
            return result['bank_id'] if result else None
        except psycopg2.Error as e:
            logger.error(f"❌ Error getting bank ID: {e}")
            return None
    
    def test_connection(self) -> bool:
        """Test database connection and basic operations"""
        try:
            # Test basic query
            self.cursor.execute("SELECT version()")
            version = self.cursor.fetchone()
            logger.info(f"✅ PostgreSQL Version: {version['version']}")
            
            # Test tables exist
            self.cursor.execute("""
                SELECT table_name FROM information_schema.tables 
                WHERE table_schema = 'public' AND table_name IN ('banks', 'reviews')
            """)
            tables = [row['table_name'] for row in self.cursor.fetchall()]
            logger.info(f"✅ Tables found: {tables}")
            
            # Test bank data
            self.cursor.execute("SELECT COUNT(*) as count FROM banks")
            bank_count = self.cursor.fetchone()['count']
            logger.info(f"✅ Banks in database: {bank_count}")
            
            # Test review data
            self.cursor.execute("SELECT COUNT(*) as count FROM reviews")
            review_count = self.cursor.fetchone()['count']
            logger.info(f"✅ Reviews in database: {review_count}")
            
            return True
            
        except psycopg2.Error as e:
            logger.error(f"❌ Database test failed: {e}")
            return False

def setup_database() -> bool:
    """Main function to set up the complete database"""
    logger.info("🚀 Starting PostgreSQL Database Setup for Bank Reviews Analysis")
    
    db = BankReviewsDB()
    
    try:
        # Step 1: Create database
        if not db.create_database():
            return False
        
        # Step 2: Connect to the database
        if not db.connect():
            return False
        
        # Step 3: Create tables
        if not db.create_tables():
            return False
        
        # Step 4: Insert initial bank data
        if not db.insert_banks_data():
            return False
        
        # Step 5: Test everything
        if not db.test_connection():
            return False
        
        logger.info("🎉 Database setup completed successfully!")
        logger.info(f"📝 Connection String: {db.get_connection_string()}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Database setup failed: {e}")
        return False
    
    finally:
        db.disconnect()

if __name__ == "__main__":
    print("=" * 60)
    print("PostgreSQL Database Setup for Bank Reviews Analysis")
    print("=" * 60)
    
    success = setup_database()
    
    if success:
        print("\n✅ SUCCESS: Database setup completed successfully!")
        print(f"🔗 Connection String: {CONNECTION_STRING.format(**DB_CONFIG)}")
        print("📚 Next steps:")
        print("   1. Run 'python data_insertion.py' to load your review data")
        print("   2. Use the provided connection string in your applications")
    else:
        print("\n❌ FAILED: Database setup encountered errors!")
        print("📚 Please check the logs and fix any issues before proceeding") 