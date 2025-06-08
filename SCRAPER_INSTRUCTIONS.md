# Ethiopian Banking Apps Review Scraper

This project scrapes customer reviews from Google Play Store for Ethiopian banking applications to analyze customer experience and sentiment.

## 📱 Apps Being Scraped

1. **Dashen Bank** - `com.dashen.dashensuperapp`
2. **Commercial Bank of Ethiopia (CBE)** - `com.combanketh.mobilebanking`
3. **Bank of Abyssinia (BOA)** - `com.boa.boaMobileBanking`

## 🎯 Features

- **Minimum 400 reviews per app**: Uses multiple strategies to collect sufficient data
- **Ethiopian market focus**: Configured for Ethiopia (`country='et'`)
- **Duplicate removal**: Automatically removes duplicate reviews
- **Multiple sort methods**: Uses both NEWEST and MOST_RELEVANT sorting
- **Structured output**: Saves data in CSV format with consistent schema
- **Logging**: Comprehensive logging for monitoring and debugging
- **Scheduling support**: Can run on schedule or one-time execution

## 🚀 How to Run the Scraper

### Option 1: One-Time Scraping (Recommended for testing)

```bash
# Navigate to your project directory
cd /path/to/your/project

# Activate virtual environment (if you have one)
.venv\Scripts\activate  # Windows
# or
source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Run the one-time scraper
python scripts/run_scraper.py
```

### Option 2: Scheduled Scraping (For continuous monitoring)

```bash
# Run the scheduled scraper (runs immediately + schedules daily at 1 AM)
python scripts/play_store_scraper.py
```

## 📊 Output Data Structure

The scraper creates CSV files in the `data/` folder with the following structure:

| Column        | Description               | Example                        |
| ------------- | ------------------------- | ------------------------------ |
| `review_text` | The actual review content | "Great app, easy to use..."    |
| `rating`      | Star rating (1-5)         | 4                              |
| `date`        | Review date               | 2024-01-15                     |
| `bank_name`   | Full bank name            | "Commercial Bank of Ethiopia"  |
| `source`      | Data source               | "Google Play"                  |
| `app_id`      | App package identifier    | "com.combanketh.mobilebanking" |

## 📁 File Structure

```
data/
├── Dashen_Bank_reviews_20240115_143022.csv
├── CBE_reviews_20240115_143045.csv
└── BOA_reviews_20240115_143108.csv

scripts/
├── play_store_scraper.py    # Scheduled scraper
└── run_scraper.py          # One-time scraper
```

## 🔧 Configuration

### Minimum Reviews Target

The scraper targets a minimum of 400 reviews per app. If fewer are available, it will:

1. Try different sorting methods (NEWEST, MOST_RELEVANT)
2. Combine results and remove duplicates
3. Log warnings if the minimum cannot be reached

### Rate Limiting

- 5-second delay between apps to respect API limits
- Uses Ethiopia country code (`'et'`) for localized results
- Requests double the minimum initially to account for restrictions

## 🐛 Troubleshooting

### Common Issues

1. **Low review count**: Some apps may have fewer than 400 reviews available

   - The scraper will collect as many as possible and log warnings
   - Check the log file for specific counts

2. **Connection errors**: Network or API issues

   - Check internet connection
   - Try running again later
   - Check `scraper.log` for detailed error messages

3. **App not found**: App ID might be incorrect or app unavailable in Ethiopia
   - Verify the app exists on Google Play Store
   - Check if app is available in Ethiopian market

### Logs

Check `scraper.log` for detailed execution logs:

```bash
tail -f scraper.log  # Follow logs in real-time
```

## 📈 Data Analysis Ready

The output CSV files are optimized for data analysis with:

- Consistent column names across all files
- Standardized date format (YYYY-MM-DD)
- UTF-8 encoding for international characters
- Ready for pandas DataFrame loading

```python
import pandas as pd

# Load scraped data
df = pd.read_csv('data/CBE_reviews_20240115_143045.csv')
print(df.head())
```

## ⚙️ Advanced Configuration

### Modify Target Apps

Edit the `BANKING_APPS` dictionary in either script:

```python
BANKING_APPS = {
    'Your_Bank': {
        'app_id': 'com.yourbank.app',
        'bank_name': 'Your Bank Name'
    }
}
```

### Change Scheduling

In `play_store_scraper.py`, uncomment different scheduling options:

```python
# schedule.every().day.at("01:00").do(scrape_all_banking_apps)  # Daily at 1 AM
# schedule.every(6).hours.do(scrape_all_banking_apps)           # Every 6 hours
# schedule.every().monday.do(scrape_all_banking_apps)           # Every Monday
# schedule.every().hour.do(scrape_all_banking_apps)             # Every hour
```

## 🤝 Support

If you encounter issues:

1. Check the logs in `scraper.log`
2. Ensure all dependencies are installed
3. Verify internet connection
4. Check that the apps exist and are available in Ethiopia

---

**Happy Scraping! 🎉**
