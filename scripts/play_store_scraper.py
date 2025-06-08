from google_play_scraper import Sort, reviews
import csv
import os
from datetime import datetime
import schedule
import logging
import time

# Set up logging
logging.basicConfig(filename='scraper.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Banking apps configuration
BANKING_APPS = {
    'Dashen Bank': {
        'app_id': 'com.dashen.dashensuperapp',
        'bank_name': 'Dashen Bank'
    },
    'CBE': {
        'app_id': 'com.combanketh.mobilebanking', 
        'bank_name': 'Commercial Bank of Ethiopia'
    },
    'BOA': {
        'app_id': 'com.boa.boaMobileBanking',
        'bank_name': 'Bank of Abyssinia'
    }
}

def ensure_data_folder():
    """Create data folder if it doesn't exist"""
    if not os.path.exists('data'):
        os.makedirs('data')
        logging.info("Created data folder")

def scrape_single_app_reviews(app_name, app_config, min_reviews=400):
    """Scrape reviews for a single app with retry mechanism to ensure minimum reviews"""
    app_id = app_config['app_id']
    bank_name = app_config['bank_name']
    
    logging.info(f"Fetching reviews for {app_name} ({app_id})...")
    
    try:
        # Try to get more reviews initially to account for potential restrictions
        initial_count = max(min_reviews * 2, 1000)  # Start with double the minimum or 1000
        
        results, _ = reviews(
            app_id,
            lang='en',
            country='et',  # Ethiopia country code
            sort=Sort.NEWEST,
            count=initial_count,
            filter_score_with=None
        )
        
        # If we got fewer than minimum, try different approaches
        if len(results) < min_reviews:
            logging.warning(f"Only got {len(results)} reviews for {app_name}, trying different sort...")
            
            # Try with different sort order
            results_most_relevant, _ = reviews(
                app_id,
                lang='en',
                country='et',
                sort=Sort.MOST_RELEVANT,
                count=initial_count,
                filter_score_with=None
            )
            
            # Combine results and remove duplicates based on review content
            combined_results = results + results_most_relevant
            unique_results = []
            seen_content = set()
            
            for review in combined_results:
                if review['content'] not in seen_content:
                    unique_results.append(review)
                    seen_content.add(review['content'])
            
            results = unique_results
        
        # Ensure we have at least the minimum number of reviews
        if len(results) < min_reviews:
            logging.warning(f"Could only retrieve {len(results)} reviews for {app_name} (minimum: {min_reviews})")
        else:
            logging.info(f"Retrieved {len(results)} reviews for {app_name}")
        
        # Save to CSV in data folder
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'data/{app_name.replace(" ", "_")}_reviews_{timestamp}.csv'
        
        ensure_data_folder()
        
        with open(filename, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=['review_text', 'rating', 'date', 'bank_name', 'source', 'app_id'])
            writer.writeheader()
            
            for entry in results:
                writer.writerow({
                    'review_text': entry['content'],
                    'rating': entry['score'],
                    'date': entry['at'].strftime('%Y-%m-%d'),
                    'bank_name': bank_name,
                    'source': 'Google Play',
                    'app_id': app_id
                })
        
        logging.info(f"Saved {len(results)} reviews for {app_name} to {filename}")
        return len(results)
        
    except Exception as e:
        logging.error(f"Error scraping {app_name}: {e}")
        return 0

def scrape_all_banking_apps():
    """Scrape reviews for all configured banking apps"""
    logging.info("Starting batch scraping for all banking apps...")
    total_reviews = 0
    
    for app_name, app_config in BANKING_APPS.items():
        review_count = scrape_single_app_reviews(app_name, app_config, min_reviews=400)
        total_reviews += review_count
        
        # Add delay between apps to be respectful to the API
        time.sleep(5)
    
    logging.info(f"Completed scraping all apps. Total reviews collected: {total_reviews}")

# Run immediately on script start
if __name__ == "__main__":
    scrape_all_banking_apps()

# Scheduling options (uncomment the one you want to use):
schedule.every().day.at("01:00").do(scrape_all_banking_apps)  # Daily at 1 AM
# schedule.every(6).hours.do(scrape_all_banking_apps)           # Every 6 hours
# schedule.every().monday.do(scrape_all_banking_apps)           # Every Monday
# schedule.every().hour.do(scrape_all_banking_apps)             # Every hour

print("🔄 Starting Ethiopian Banking Apps Review Scraper...")
print("📱 Apps to scrape:")
for app_name, config in BANKING_APPS.items():
    print(f"   • {app_name}: {config['app_id']}")
print("📂 Results will be saved to 'data/' folder")
print("⏰ Scheduled to run daily at 1:00 AM")
print("🛑 Press Ctrl+C to stop the scheduler")

while True:
    schedule.run_pending()
    time.sleep(1)