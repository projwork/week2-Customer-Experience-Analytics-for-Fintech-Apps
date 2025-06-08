# Ethiopian Banking Apps Customer Experience Analysis

A comprehensive project for scraping, preprocessing, and analyzing customer reviews from Google Play Store for Ethiopian banking applications to understand customer experience patterns and provide actionable insights.

## 🏦 Supported Banks

- **CBE** (Commercial Bank of Ethiopia): `com.combanketh.mobilebanking`
- **BOA** (Bank of Abyssinia): `com.boa.boaMobileBanking`
- **Dashen** (Dashen Bank): `com.dashen.dashensuperapp`

## 🚀 Features

### Data Collection

- **Multi-bank scraping**: Automated scraping from CBE, BOA, and Dashen Bank apps
- **Target achievement**: 1200+ reviews collected (400+ per bank)
- **Ethiopian market focus**: Configured specifically for Ethiopian Play Store
- **Rate limiting protection**: Smart delays and retry logic
- **Duplicate prevention**: Automatic deduplication across sorting methods

### Data Preprocessing

- **Comprehensive cleaning**: Text normalization, date standardization
- **Missing data handling**: <5% missing data target achieved
- **Quality validation**: Data quality reports and metrics
- **Feature engineering**: Review length, sentiment categories, temporal features

### Analysis Ready

- **Jupyter notebook**: Interactive analysis environment
- **Visualization tools**: Matplotlib, Seaborn, Plotly integration
- **Statistical analysis**: Customer satisfaction patterns and trends
- **Comparative insights**: Cross-bank performance analysis

## 📁 Project Structure

```
├── scripts/
│   ├── play_store_scraper.py    # Scheduled scraper script
│   ├── run_scraper.py          # One-time scraper script
│   └── data_preprocessing.py   # Data cleaning & preprocessing
├── data/
│   ├── raw/                    # Raw scraped data
│   └── processed/              # Clean, analysis-ready data
├── notebooks/
│   └── fintechcustomerExperience.ipynb  # Analysis notebook
├── requirements.txt            # Python dependencies
├── SCRAPER_INSTRUCTIONS.md     # Detailed scraping guide
└── README.md                   # This file
```

## 📋 Prerequisites

- Python 3.8 or higher
- Internet connection
- Virtual environment (recommended)

## 🛠️ Installation

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd FIntech-Customer-Experience
```

### 2. Create and activate virtual environment

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux/Mac
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

## 📊 Methodology

### 1. Data Collection Strategy

**Target**: 1,200+ customer reviews (400+ per bank)
**Source**: Google Play Store (Ethiopian market)
**Period**: Most recent reviews available

#### Banking Apps Analyzed:

- **CBE Mobile Banking** (`com.combanketh.mobilebanking`)
- **BOA Mobile Banking** (`com.boa.boaMobileBanking`)
- **Dashen SuperApp** (`com.dashen.dashensuperapp`)

#### Collection Approach:

1. **Multi-sort strategy**: NEWEST + MOST_RELEVANT sorting
2. **Deduplication**: Remove identical content across sorting methods
3. **Rate limiting**: 5-second delays between banks, respectful API usage
4. **Quality assurance**: Minimum review thresholds with retry logic

### 2. Data Preprocessing Pipeline

#### Text Cleaning:

- **Normalization**: Whitespace standardization, character encoding
- **Language handling**: Support for English, Amharic, and mixed content
- **Special characters**: Preserve essential punctuation, remove noise

#### Data Standardization:

- **Dates**: Convert to YYYY-MM-DD format
- **Ratings**: Validate 1-5 star scale, handle outliers
- **Bank names**: Standardize to consistent format (CBE, BOA, Dashen Bank)

#### Quality Control:

- **Missing data**: Target <5% missing across all fields
- **Duplicates**: Content-based deduplication within and across banks
- **Validation**: Schema validation and data type consistency

### 3. Feature Engineering

#### Derived Features:

- **Review length**: Character and word count metrics
- **Sentiment categories**: Negative (1-2), Neutral (3), Positive (4-5)
- **Temporal features**: Year, month, quarter for trend analysis
- **Quality flags**: Valid rating indicator, data completeness scores

#### Output Schema:

```
review, rating, date, bank, source, review_length, word_count,
rating_category, year, month, quarter, rating_valid
```

### 4. Analysis Framework

#### Key Performance Indicators (KPIs):

- **Data completeness**: <5% missing data achieved
- **Sample size**: 1,200+ reviews collected
- **Coverage**: All three major banks represented
- **Temporal scope**: Recent customer feedback patterns

#### Quality Metrics:

- **Duplicate rate**: Percentage of reviews removed
- **Missing data rate**: Per field and overall
- **Review distribution**: Balanced representation across banks
- **Date coverage**: Temporal distribution of feedback

### 5. Validation & Verification

#### Data Quality Checks:

- ✅ Schema validation for all required fields
- ✅ Rating range validation (1-5 stars)
- ✅ Date format consistency (YYYY-MM-DD)
- ✅ Text encoding verification (UTF-8)
- ✅ Bank name standardization

#### Statistical Validation:

- ✅ Sample size sufficiency (400+ per bank)
- ✅ Missing data threshold (<5%)
- ✅ Duplicate removal verification
- ✅ Cross-bank comparison feasibility

## 🎯 Usage

### Method 1: Using the Runner Script (Recommended)

#### Show bank information

```bash
cd scripts
python run_scraper.py --info
```

#### Run scraping once for all banks

```bash
cd scripts
python run_scraper.py --once
```

#### Run with scheduling (daily at 1 AM)

```bash
cd scripts
python run_scraper.py --schedule
```

### Method 2: Direct Script Execution

#### Run once

```bash
cd scripts
python play_store_scraper.py
```

#### Run with custom scheduling

Edit the `play_store_scraper.py` file and uncomment your preferred schedule option:

```python
# Schedule options (uncomment the one you want to use):
schedule.every().day.at("01:00").do(scrape_all_banks)  # Daily at 1 AM
# schedule.every(12).hours.do(scrape_all_banks)          # Every 12 hours
# schedule.every().monday.do(scrape_all_banks)           # Every Monday
# schedule.every(6).hours.do(scrape_all_banks)           # Every 6 hours
```

## 🗃️ Output Data Structure

The scraper saves data to CSV files in the `data/` folder with the following structure:

| Field             | Description                  |
| ----------------- | ---------------------------- |
| `review_id`       | Unique review identifier     |
| `review_text`     | The actual review content    |
| `rating`          | Star rating (1-5)            |
| `date`            | Review date and time         |
| `bank_name`       | Full bank name               |
| `bank_code`       | Bank code (CBE, BOA, Dashen) |
| `source`          | "Google Play Store"          |
| `thumbs_up_count` | Number of helpful votes      |
| `reviewer_name`   | Reviewer's username          |

### Example output files:

- `data/CBE_reviews_20241201_143022.csv`
- `data/BOA_reviews_20241201_143055.csv`
- `data/Dashen_reviews_20241201_143128.csv`

## 🔧 Configuration

### Adding New Banks

Edit the `BANKS_CONFIG` in `play_store_scraper.py`:

```python
BANKS_CONFIG = {
    'NEW_BANK': {
        'app_id': 'com.newbank.app',
        'bank_name': 'New Bank Name',
        'min_reviews': 400
    }
}
```

### Adjusting Minimum Reviews

Change the `min_reviews` value for any bank in the configuration.

### Modifying Schedule

Edit the schedule options in the `run_scheduled_scraper()` function.

## 🚨 Important Notes

### Google Play Store Limitations

- Google Play Store has rate limiting
- The scraper includes built-in delays and retry logic
- If you encounter persistent errors, increase the delays between requests

### Data Quality

- The scraper automatically removes duplicates
- Some apps may have fewer than 400 reviews available
- Reviews are collected in multiple passes using different sorting methods

### Rate Limiting Protection

- Random delays between requests (2-7 seconds)
- Exponential backoff on failures
- 30-second delay between different banks

## 📊 Monitoring

### Log Files

- Logs are saved to `logs/scraper.log`
- Console output shows real-time progress
- Each scraping session includes a summary report

### Success Indicators

- ✅ Successful data collection
- 📊 Number of reviews collected
- 💾 File saved confirmation
- 📊 Final summary with all banks

## 🐛 Troubleshooting

### Common Issues

1. **"No reviews found"**

   - Check if the app ID is correct
   - Some apps may have very few reviews
   - Try running again later

2. **Rate limiting errors**

   - The scraper handles this automatically
   - If persistent, increase delays in the script

3. **ImportError: No module named 'google_play_scraper'**

   - Make sure you've installed dependencies: `pip install -r requirements.txt`
   - Ensure your virtual environment is activated

4. **Permission errors writing files**
   - Check that you have write permissions in the project directory
   - Ensure the `data/` and `logs/` directories exist

### Getting Help

- Check the log files in `logs/scraper.log`
- Enable debug logging by changing the log level in the script
- Ensure all dependencies are correctly installed

## 📈 Data Analysis

After scraping, your CSV files in the `data/` folder are ready for analysis with:

- Pandas for data manipulation
- Jupyter notebooks for exploratory analysis
- Sentiment analysis libraries
- Data visualization tools

## 🔒 Ethical Considerations

- This scraper respects Google Play Store's rate limits
- Data is collected for legitimate research/analysis purposes
- No personal information beyond public reviews is collected
- Follow Google's Terms of Service when using this tool

## 📞 Support

For issues, questions, or contributions, please refer to the project documentation or create an issue in the repository.

---

**Happy Scraping! 🎉**
