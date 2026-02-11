"""
Selenium-based Scraper for Google Issue Tracker
================================================
This script reuses the analysis logic from main.py but uses Selenium 
for web scraping to handle potential blocking or SSL issues.
"""

import logging
import time
import sys
import pandas as pd
from tqdm import tqdm
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup

# Import helper functions from main.py
# ensure main.py is in the same directory
try:
    from main import (
        setup_directories, setup_logging, load_csv, remove_duplicates,
        detect_url_column, construct_urls_from_ids, apply_categorization,
        create_bar_chart, create_pie_chart, create_summary_table, export_results
    )
except ImportError:
    print("Error: main.py not found or could not import functions.")
    sys.exit(1)

import os

def setup_selenium_driver():
    """
    Setup Chrome driver with appropriate options.
    """
    # Disable SSL verify for webdriver manager
    os.environ['WDM_SSL_VERIFY'] = '0'
    
    chrome_options = Options()
    chrome_options.add_argument("--headless")  # Run properly without UI
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--ignore-certificate-errors")
    chrome_options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")
    
    # Mute selenium logs
    logging.getLogger('selenium').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)
    return driver

def scrape_with_selenium(df, url_column, logger):
    """
    Scrape issues using Selenium.
    """
    logger.info("Starting Selenium web scraping...")
    
    descriptions = []
    labels_list = []
    successful_scrapes = 0
    failed_scrapes = 0
    
    driver = None
    try:
        driver = setup_selenium_driver()
        logger.info("[SUCCESS] Selenium driver initialized")
    except Exception as e:
        logger.error(f"[ERROR] Failed to initialize Selenium driver: {e}")
        # Initialize empty columns to prevent downstream errors
        df['scraped_description'] = "Scraping Failed"
        df['scraped_labels'] = "N/A"
        return df, 0, 0

    # Progress bar for scraping
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Scraping issues (Selenium)"):
        url = row[url_column]
        
        # Validate URL
        if pd.isna(url) or not isinstance(url, str):
            descriptions.append("Invalid URL")
            labels_list.append("N/A")
            failed_scrapes += 1
            continue
        
        # Add https:// if missing
        if not url.startswith('http'):
            url = 'https://' + url
            
        try:
            driver.get(url)
            
            # Wait briefly for page load (implicit wait handles most, but a small sleep helps with dynamic content render)
            # Better to use explicit wait if possible, but structure allows dynamic load
            time.sleep(2) 
            
            # We can use BeautifulSoup on the driver.page_source for easier parsing
            # equivalent to the original logic
            soup = BeautifulSoup(driver.page_source, 'lxml')
            
            # Extract issue description (reusing logic from main.py's logic effectively)
            description = ""
            # Adjust selectors if needed, but the original ones seemed correct for the structure
            # main.py used: desc_elements = soup.find_all(['p', 'div'], class_=re.compile('description|issue-desc|comment'))
            # We'll try to map that or just use the same BS4 logic on the rendered page
            
            import re
            desc_elements = soup.find_all(['p', 'div'], class_=re.compile('description|issue-desc|comment'))
            if desc_elements:
                description = " ".join([elem.get_text(strip=True) for elem in desc_elements[:3]])
            
            # Extract labels/tags
            labels = []
            label_elements = soup.find_all(['span', 'div'], class_=re.compile('label|tag|chip'))
            if label_elements:
                labels = [elem.get_text(strip=True) for elem in label_elements]
            
            # If description is empty, try Selenium placeholders just in case specific JS didn't load for BS4
            if not description:
                try:
                    # Fallback generic locator for content
                    body_text = driver.find_element(By.TAG_NAME, "body").text
                    description = body_text[:500] if body_text else "No description available"
                except:
                    pass

            descriptions.append(description[:500] if description else "No description available")
            labels_list.append(', '.join(labels) if labels else "No labels")
            successful_scrapes += 1
            
        except Exception as e:
            # logger.warning(f"Failed to scrape {url}: {e}") # specific log if needed
            descriptions.append("Scraping failed")
            labels_list.append("N/A")
            failed_scrapes += 1

    if driver:
        driver.quit()
    
    df['scraped_description'] = descriptions
    df['scraped_labels'] = labels_list
    
    logger.info(f"[SUCCESS] Scraping complete: {successful_scrapes} successful, {failed_scrapes} failed")
    return df, successful_scrapes, failed_scrapes

def main():
    # Setup directories and logging
    setup_directories()
    logger = setup_logging()
    
    logger.info("Running Selenium-based Scraper Version")

    # Load CSV
    # Explicitly using the file requested by user
    input_file = 'input/issueTracker 2(in) (1).csv'
    logger.info(f"Loading input file: {input_file}")
    
    df = load_csv(input_file, logger)
    if df is None or len(df) == 0:
        logger.error("Failed to load data or empty file.")
        return
        
    original_count = len(df)
    
    # Pipeline steps
    df = remove_duplicates(df, logger)
    duplicates_removed = original_count - len(df)
    
    url_column = detect_url_column(df, logger)
    if url_column == 'ID':
        df = construct_urls_from_ids(df, 'ID', logger)
        url_column = 'url'
    elif url_column is None:
        logger.error("No URL/ID column found.")
        return

    # Scrape with Selenium
    df, successful, failed = scrape_with_selenium(df, url_column, logger)
    
    # Analysis & Export
    df = apply_categorization(df, logger)
    
    create_bar_chart(df, logger)
    create_pie_chart(df, logger)
    summary_df = create_summary_table(df, logger)
    
    stats = {
        'total_issues': original_count,
        'successful_scrapes': successful,
        'failed_scrapes': failed,
        'duplicates_removed': duplicates_removed
    }
    
    export_results(df, summary_df, logger, stats)
    logger.info("Selenium Run Completed.")

if __name__ == "__main__":
    main()
