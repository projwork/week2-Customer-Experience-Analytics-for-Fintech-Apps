# PostgreSQL Setup Guide for Bank Reviews Database

## Overview

This guide will help you set up PostgreSQL locally and configure the `bank_reviews` database for the Ethiopian Banking Apps Review Analysis project.

## Prerequisites

- Windows 10/11
- Administrative privileges
- Internet connection for downloads

## Step 1: Install PostgreSQL

### Option A: Download from Official Website

1. Visit [PostgreSQL Downloads](https://www.postgresql.org/download/windows/)
2. Download PostgreSQL 15+ for Windows
3. Run the installer as Administrator
4. During installation:
   - Set the password for the `postgres` superuser (remember this!)
   - Use default port `5432`
   - Accept default data directory
   - Install all components (PostgreSQL Server, pgAdmin, Command Line Tools)

### Option B: Using Chocolatey (if available)

```powershell
choco install postgresql --params '/Password:YourStrongPassword'
```

## Step 2: Verify Installation

1. Open Command Prompt as Administrator
2. Test PostgreSQL service:

```cmd
sc query postgresql-x64-15
```

3. Test psql command line:

```cmd
psql -U postgres -h localhost
```

Enter your password when prompted.

## Step 3: Configure PostgreSQL

### Check PostgreSQL Status

```cmd
# Check if PostgreSQL is running
net start | findstr postgres

# Start PostgreSQL if not running
net start postgresql-x64-15
```

### pgAdmin Setup (Optional but Recommended)

1. Open pgAdmin 4 from Start Menu
2. Create a master password
3. Connect to local server using `postgres` user and your password

## Step 4: Install Python Dependencies

```bash
pip install psycopg2-binary pandas numpy logging
```

## Step 5: Database Configuration

### Update Database Credentials

Edit the database configuration in your Python scripts:

```python
DB_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'database': 'bank_reviews',
    'user': 'postgres',
    'password': 'YOUR_ACTUAL_PASSWORD'  # Change this!
}
```

### Connection String Format

```
postgresql://postgres:YOUR_PASSWORD@localhost:5432/bank_reviews
```

## Step 6: Run Database Setup

1. Navigate to your project scripts directory:

```cmd
cd C:\Kifiya\Week2\FIntech-Customer-Experience\scripts
```

2. Run the database setup:

```cmd
python database_setup.py
```

3. Load your data:

```cmd
python data_insertion.py
```

## Database Schema

### Banks Table

```sql
CREATE TABLE banks (
    bank_id SERIAL PRIMARY KEY,
    bank_name VARCHAR(255) NOT NULL UNIQUE,
    bank_code VARCHAR(10) NOT NULL UNIQUE,
    app_id VARCHAR(255) NOT NULL UNIQUE,
    full_name VARCHAR(255) NOT NULL,
    country VARCHAR(100) DEFAULT 'Ethiopia',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### Reviews Table

```sql
CREATE TABLE reviews (
    review_id SERIAL PRIMARY KEY,
    bank_id INTEGER NOT NULL REFERENCES banks(bank_id),
    review_text TEXT NOT NULL,
    rating INTEGER NOT NULL CHECK (rating >= 1 AND rating <= 5),
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
```

## Sample Queries

### Check Database Status

```sql
-- Count total reviews
SELECT COUNT(*) as total_reviews FROM reviews;

-- Reviews by bank
SELECT b.bank_name, COUNT(r.review_id) as review_count
FROM banks b
LEFT JOIN reviews r ON b.bank_id = r.bank_id
GROUP BY b.bank_name;

-- Rating distribution
SELECT rating, COUNT(*) as count
FROM reviews
GROUP BY rating
ORDER BY rating;
```

### Advanced Analytics Queries

```sql
-- Average rating by bank
SELECT b.bank_name,
       AVG(r.rating) as avg_rating,
       COUNT(r.review_id) as total_reviews
FROM banks b
JOIN reviews r ON b.bank_id = r.bank_id
GROUP BY b.bank_name;

-- Monthly review trends
SELECT year, month, COUNT(*) as review_count
FROM reviews
GROUP BY year, month
ORDER BY year, month;

-- Sentiment distribution by bank
SELECT b.bank_name,
       r.rating_category,
       COUNT(*) as count
FROM banks b
JOIN reviews r ON b.bank_id = r.bank_id
GROUP BY b.bank_name, r.rating_category;
```

## Troubleshooting

### Common Issues and Solutions

#### 1. PostgreSQL Service Not Starting

```cmd
# Check Windows services
services.msc

# Restart PostgreSQL service
net stop postgresql-x64-15
net start postgresql-x64-15
```

#### 2. Connection Refused

- Verify PostgreSQL is running
- Check firewall settings
- Ensure correct port (5432)
- Verify credentials

#### 3. psycopg2 Installation Issues

```bash
# Try binary version
pip install psycopg2-binary

# Or compile from source
pip install psycopg2
```

#### 4. Permission Denied

- Run Command Prompt as Administrator
- Check PostgreSQL data directory permissions
- Verify user has database access

#### 5. Password Authentication Failed

- Double-check your password
- Reset postgres password if needed:

```cmd
psql -U postgres
ALTER USER postgres PASSWORD 'new_password';
```

## Performance Tuning

### Optimize for Large Datasets

```sql
-- Analyze tables for better query performance
ANALYZE banks;
ANALYZE reviews;

-- Update statistics
VACUUM ANALYZE;
```

### Useful Configuration

Edit `postgresql.conf` for better performance:

```
shared_buffers = 256MB
effective_cache_size = 1GB
maintenance_work_mem = 64MB
```

## Backup and Restore

### Create Backup

```cmd
pg_dump -U postgres -h localhost bank_reviews > bank_reviews_backup.sql
```

### Restore from Backup

```cmd
psql -U postgres -h localhost bank_reviews < bank_reviews_backup.sql
```

## Security Best Practices

1. **Change Default Password**: Never use 'password' in production
2. **Limit Connections**: Configure `pg_hba.conf` for restricted access
3. **Use SSL**: Enable SSL for encrypted connections
4. **Regular Updates**: Keep PostgreSQL updated
5. **Backup Strategy**: Implement regular automated backups

## Next Steps

After successful setup:

1. Verify >1,000 reviews are loaded
2. Test connection string in your applications
3. Run sample queries to validate data
4. Set up regular backup schedule
5. Consider monitoring tools like pgAdmin or DBeaver

## Contact and Support

For issues with this setup:

1. Check PostgreSQL official documentation
2. Review error logs in PostgreSQL data directory
3. Use pgAdmin for visual database management
4. Consider PostgreSQL community forums for advanced issues

---

**Connection String for Your Applications:**

```
postgresql://postgres:YOUR_PASSWORD@localhost:5432/bank_reviews
```

Remember to replace `YOUR_PASSWORD` with your actual PostgreSQL password!
