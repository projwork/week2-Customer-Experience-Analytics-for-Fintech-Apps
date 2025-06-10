#!/usr/bin/env python3
"""
PostgreSQL Database Setup for Bank Reviews
"""

import psycopg2
import psycopg2.extras
import logging

# Configuration
DB_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'database': 'bank_reviews',
    'user': 'postgres',
    'password': 'password'  # CHANGE THIS!
}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_database():
    """Create the bank_reviews database"""
    try:
        # Connect to default postgres database
        config = DB_CONFIG.copy()
        config['database'] = 'postgres'
        
        conn = psycopg2.connect(**config)
        conn.autocommit = True
        cursor = conn.cursor()
        
        # Check if database exists
        cursor.execute("SELECT 1 FROM pg_database WHERE datname = %s", ('bank_reviews',))
        if not cursor.fetchone():
            cursor.execute("CREATE DATABASE bank_reviews")
            logger.info("Database 'bank_reviews' created successfully")
        else:
            logger.info("Database 'bank_reviews' already exists")
        
        cursor.close()
        conn.close()
        return True
    except Exception as e:
        logger.error(f"Error creating database: {e}")
        return False

def create_tables():
    """Create tables in the database"""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()
        
        # Banks table
        banks_sql = """
        CREATE TABLE IF NOT EXISTS banks (
            bank_id SERIAL PRIMARY KEY,
            bank_name VARCHAR(255) NOT NULL UNIQUE,
            bank_code VARCHAR(10) NOT NULL UNIQUE,
            app_id VARCHAR(255) NOT NULL UNIQUE,
            full_name VARCHAR(255) NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
        
        # Reviews table
        reviews_sql = """
        CREATE TABLE IF NOT EXISTS reviews (
            review_id SERIAL PRIMARY KEY,
            bank_id INTEGER REFERENCES banks(bank_id),
            review_text TEXT NOT NULL,
            rating INTEGER CHECK (rating >= 1 AND rating <= 5),
            review_date DATE NOT NULL,
            source VARCHAR(100) DEFAULT 'Google Play',
            review_length INTEGER,
            word_count INTEGER,
            rating_category VARCHAR(20),
            year INTEGER,
            month INTEGER,
            quarter INTEGER,
            rating_valid BOOLEAN DEFAULT TRUE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
        
        cursor.execute(banks_sql)
        cursor.execute(reviews_sql)
        
        # Insert bank data
        insert_banks = """
        INSERT INTO banks (bank_name, bank_code, app_id, full_name) VALUES
        ('CBE', 'CBE', 'com.combanketh.mobilebanking', 'Commercial Bank of Ethiopia'),
        ('BOA', 'BOA', 'com.boa.boaMobileBanking', 'Bank of Abyssinia'),
        ('Dashen Bank', 'DASH', 'com.dashen.dashensuperapp', 'Dashen Bank')
        ON CONFLICT (bank_name) DO NOTHING;
        """
        
        cursor.execute(insert_banks)
        conn.commit()
        
        logger.info("Tables and data created successfully")
        
        cursor.close()
        conn.close()
        return True
        
    except Exception as e:
        logger.error(f"Error creating tables: {e}")
        return False

def main():
    print("Setting up PostgreSQL database for Bank Reviews...")
    
    if create_database() and create_tables():
        print("✅ Database setup completed successfully!")
        print(f"🔗 Connection string: postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}")
    else:
        print("❌ Database setup failed!")

if __name__ == "__main__":
    main() 