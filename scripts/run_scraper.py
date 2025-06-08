#!/usr/bin/env python3
"""
Simple script to run the banking app review scraper once
Usage: python scripts/run_scraper.py
"""

from google_play_scraper import Sort, reviews
import csv
import os
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('scraper.log'),
        logging.StreamHandler()  # Also log to console
    ]
)

# Banking apps configuration
BANKING_APPS = {
    'Dashen_Bank': {
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
    
    print(f"\n🔄 Fetching reviews for {app_name} ({app_id})...")
    logging.info(f"Fetching reviews for {app_name} ({app_id})...")
    
    try:
        # Try to get more reviews initially to account for potential restrictions
        initial_count = max(min_reviews * 2, 1000)  # Start with double the minimum or 1000
        
        print(f"   Requesting {initial_count} reviews (target minimum: {min_reviews})...")
        
        results, _ = reviews(
            app_id,
            lang='en',
            country='et',  # Ethiopia country code
            sort=Sort.NEWEST,
            count=initial_count,
            filter_score_with=None
        )
        
        print(f"   Got {len(results)} reviews with NEWEST sort")
        
        # If we got fewer than minimum, try different approaches
        if len(results) < min_reviews:
            print(f"   ⚠️ Only got {len(results)} reviews, trying MOST_RELEVANT sort...")
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
            
            print(f"   Got {len(results_most_relevant)} additional reviews with MOST_RELEVANT sort")
            
            # Combine results and remove duplicates based on review content
            combined_results = results + results_most_relevant
            unique_results = []
            seen_content = set()
            
            for review in combined_results:
                if review['content'] not in seen_content:
                    unique_results.append(review)
                    seen_content.add(review['content'])
            
            results = unique_results
            print(f"   After removing duplicates: {len(results)} unique reviews")
        
        # Ensure we have at least the minimum number of reviews
        if len(results) < min_reviews:
            print(f"   ⚠️ WARNING: Could only retrieve {len(results)} reviews (minimum: {min_reviews})")
            logging.warning(f"Could only retrieve {len(results)} reviews for {app_name} (minimum: {min_reviews})")
        else:
            print(f"   ✅ Successfully retrieved {len(results)} reviews")
            logging.info(f"Retrieved {len(results)} reviews for {app_name}")
        
        # Save to CSV in data folder
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'data/{app_name}_reviews_{timestamp}.csv'
        
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
        
        print(f"   💾 Saved to: {filename}")
        logging.info(f"Saved {len(results)} reviews for {app_name} to {filename}")
        return len(results)
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        logging.error(f"Error scraping {app_name}: {e}")
        return 0

def main():
    """Main function to scrape all banking apps"""
    print("🚀 Ethiopian Banking Apps Review Scraper")
    print("=" * 50)
    print("📱 Apps to scrape:")
    for app_name, config in BANKING_APPS.items():
        print(f"   • {app_name}: {config['app_id']}")
    print("📂 Results will be saved to 'data/' folder")
    print("=" * 50)
    
    logging.info("Starting batch scraping for all banking apps...")
    total_reviews = 0
    successful_apps = 0
    
    for i, (app_name, app_config) in enumerate(BANKING_APPS.items(), 1):
        print(f"\n[{i}/{len(BANKING_APPS)}] Processing {app_name}...")
        review_count = scrape_single_app_reviews(app_name, app_config, min_reviews=400)
        
        if review_count > 0:
            successful_apps += 1
            total_reviews += review_count
        
        # Add delay between apps to be respectful to the API
        if i < len(BANKING_APPS):  # Don't wait after the last app
            print("   ⏳ Waiting 5 seconds before next app...")
            import time
            time.sleep(5)
    
    print("\n" + "=" * 50)
    print("🎉 SCRAPING COMPLETED!")
    print(f"✅ Successfully scraped {successful_apps}/{len(BANKING_APPS)} apps")
    print(f"📊 Total reviews collected: {total_reviews}")
    print(f"📂 Check the 'data/' folder for CSV files")
    print("=" * 50)
    
    logging.info(f"Completed scraping all apps. Total reviews collected: {total_reviews}")

if __name__ == "__main__":
    main() 