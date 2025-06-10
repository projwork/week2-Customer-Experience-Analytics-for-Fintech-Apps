#!/usr/bin/env python3
"""
Test PostgreSQL Database Connection
"""

import psycopg2
import psycopg2.extras

# Configuration - UPDATE YOUR PASSWORD!
DB_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'database': 'bank_reviews',
    'user': 'postgres',
    'password': 'password'  # CHANGE THIS TO YOUR ACTUAL PASSWORD!
}

def test_connection():
    """Test database connection and basic operations"""
    try:
        print("🔍 Testing PostgreSQL connection...")
        
        # Test connection
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        
        print("✅ Connection successful!")
        
        # Test PostgreSQL version
        cursor.execute("SELECT version()")
        version = cursor.fetchone()['version']
        print(f"📊 PostgreSQL Version: {version}")
        
        # Test database exists
        cursor.execute("SELECT current_database()")
        db_name = cursor.fetchone()['current_database']
        print(f"📂 Connected to database: {db_name}")
        
        # Test tables exist
        cursor.execute("""
            SELECT table_name FROM information_schema.tables 
            WHERE table_schema = 'public' AND table_name IN ('banks', 'reviews')
        """)
        tables = [row['table_name'] for row in cursor.fetchall()]
        print(f"📋 Tables found: {tables}")
        
        if 'banks' in tables:
            cursor.execute("SELECT COUNT(*) as count FROM banks")
            bank_count = cursor.fetchone()['count']
            print(f"🏦 Banks in database: {bank_count}")
            
            # Show bank details
            cursor.execute("SELECT bank_name, app_id FROM banks")
            banks = cursor.fetchall()
            for bank in banks:
                print(f"   • {bank['bank_name']}: {bank['app_id']}")
        
        if 'reviews' in tables:
            cursor.execute("SELECT COUNT(*) as count FROM reviews")
            review_count = cursor.fetchone()['count']
            print(f"💬 Reviews in database: {review_count:,}")
            
            if review_count >= 1000:
                print("✅ KPI MET: Database contains >1,000 reviews")
            else:
                print(f"⚠️ KPI NOT MET: Only {review_count} reviews found")
        
        cursor.close()
        conn.close()
        
        # Show connection string
        print(f"\n🔗 Connection String:")
        print(f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}")
        
        return True
        
    except psycopg2.OperationalError as e:
        print(f"❌ Connection failed: {e}")
        print("\n🔧 Troubleshooting steps:")
        print("1. Ensure PostgreSQL is installed and running")
        print("2. Check if the service is started: net start postgresql-x64-15")
        print("3. Verify your password in the DB_CONFIG")
        print("4. Make sure the database 'bank_reviews' exists")
        return False
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    print("=" * 50)
    print("PostgreSQL Database Connection Test")
    print("=" * 50)
    
    if test_connection():
        print("\n✅ All tests passed! Database is ready for use.")
    else:
        print("\n❌ Tests failed. Please fix the issues above.")
        print("📚 Check the PostgreSQL_Setup_README.md for help.") 