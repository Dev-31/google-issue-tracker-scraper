"""
Google Issue Tracker Scraper & Analyzer
========================================
This script performs end-to-end data extraction, cleaning, categorization,
analysis, and visualization of Google Issue Tracker issues from CSV input.

Author: Expert Python Developer
Version: 1.2 - Fixed encoding issues for Windows
"""

import os
import sys
import subprocess
import re
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
import concurrent.futures
import urllib3
import matplotlib.pyplot as plt
import seaborn as sns

# Suppress SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


# ============================================================================
# SECTION 1: DEPENDENCY CHECKER & AUTO-INSTALLER
# ============================================================================

def check_and_install_dependencies():
    """
    Check if required packages are installed and install missing ones.
    
    This function reads requirements.txt and ensures all dependencies
    are available before running the main script.
    """
    print("=" * 70)
    print("CHECKING DEPENDENCIES")
    print("=" * 70)
    
    required_packages = [
        'pandas', 'requests', 'beautifulsoup4', 'tqdm', 
        'matplotlib', 'seaborn', 'lxml'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"[OK] {package} is installed")
        except ImportError:
            print(f"[MISSING] {package} is NOT installed")
            missing_packages.append(package)
    
    if missing_packages:
        print("\n[WARNING] Missing packages detected. Installing...")
        try:
            subprocess.check_call([
                sys.executable, '-m', 'pip', 'install', 
                '-r', 'requirements.txt'
            ])
            print("[SUCCESS] All dependencies installed successfully!")
        except subprocess.CalledProcessError:
            print("[ERROR] Failed to install dependencies. Please run:")
            print("  pip install -r requirements.txt")
            sys.exit(1)
    else:
        print("\n[SUCCESS] All dependencies are installed!\n")


# ============================================================================
# SECTION 2: SETUP DIRECTORIES & LOGGING
# ============================================================================

def setup_directories():
    """
    Create necessary directories for input, output, charts, and logs.
    """
    directories = [
        'input',
        'output',
        'output/charts',
        'output/logs'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    print("[OK] Directory structure created")


def setup_logging():
    """
    Configure logging to write to both console and file.
    Fixed for Windows encoding issues.
    
    Returns:
        logging.Logger: Configured logger instance
    """
    log_filename = f"output/logs/run_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    
    # Create file handler with UTF-8 encoding
    file_handler = logging.FileHandler(log_filename, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    
    # Create console handler with UTF-8 encoding for Windows
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    
    # Force UTF-8 encoding on Windows
    if sys.platform == 'win32':
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except:
            pass
    
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    
    # Setup logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    logger.info("=" * 70)
    logger.info("GOOGLE ISSUE TRACKER SCRAPER - NEW RUN")
    logger.info("=" * 70)
    
    return logger


# ============================================================================
# SECTION 3: CSV LOADING & CLEANING
# ============================================================================

def load_csv(file_path, logger):
    """
    Load CSV file containing issue URLs with automatic encoding detection.
    
    Args:
        file_path (str): Path to the input CSV file
        logger (logging.Logger): Logger instance
    
    Returns:
        pd.DataFrame: Loaded dataframe or None if failed
    """
    # List of encodings to try
    encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252', 'windows-1252']
    
    df = None
    successful_encoding = None
    
    for encoding in encodings:
        try:
            df = pd.read_csv(file_path, encoding=encoding)
            successful_encoding = encoding
            logger.info(f"[SUCCESS] CSV loaded with encoding: {encoding}")
            logger.info(f"[SUCCESS] Total rows loaded: {len(df)}")
            break
        except UnicodeDecodeError:
            logger.warning(f"[WARNING] Failed to load with encoding: {encoding}")
            continue
        except FileNotFoundError:
            logger.error(f"[ERROR] CSV file not found: {file_path}")
            logger.error(f"[INFO] Please place your CSV file at: {os.path.abspath(file_path)}")
            return None
        except Exception as e:
            logger.warning(f"[WARNING] Error with encoding {encoding}: {str(e)}")
            continue
    
    if df is None:
        logger.error("[ERROR] Could not load CSV with any known encoding")
        logger.error("[INFO] Please save your CSV as UTF-8 and try again")
        return None
    
    # Display column names to help user identify URL column
    logger.info(f"[INFO] Columns found: {list(df.columns)}")
    
    # Display first few rows for debugging
    logger.info(f"[INFO] First row sample: {df.iloc[0].to_dict() if len(df) > 0 else 'No data'}")
    
    return df


def remove_duplicates(df, logger):
    """
    Remove duplicate issues based on title, ID, or description.
    
    Args:
        df (pd.DataFrame): Input dataframe
        logger (logging.Logger): Logger instance
    
    Returns:
        pd.DataFrame: Cleaned dataframe with duplicate_count column
    """
    original_count = len(df)
    
    # Determine which column to use for duplicate detection
    duplicate_column = None
    
    if 'ID' in df.columns:
        duplicate_column = 'ID'
    elif 'Title' in df.columns:
        duplicate_column = 'Title'
    elif 'title' in df.columns:
        duplicate_column = 'title'
    elif 'description' in df.columns:
        duplicate_column = 'description'
    
    if duplicate_column:
        # Count duplicates before removal
        duplicate_counts = df.groupby(duplicate_column).size()
        df['duplicate_count'] = df[duplicate_column].map(duplicate_counts)
        
        # Remove duplicates
        df = df.drop_duplicates(subset=[duplicate_column], keep='first')
        logger.info(f"[INFO] Using '{duplicate_column}' column for duplicate detection")
    else:
        df['duplicate_count'] = 1
        logger.info("[INFO] No suitable column found for duplicate detection, keeping all rows")
    
    removed_count = original_count - len(df)
    logger.info(f"[SUCCESS] Duplicates removed: {removed_count}")
    logger.info(f"[SUCCESS] Remaining unique issues: {len(df)}")
    
    return df.reset_index(drop=True)


# ============================================================================
# SECTION 4: WEB SCRAPING
# ============================================================================

def scrape_issue_page(url, logger):
    """
    Scrape a single Google Issue Tracker page to extract issue details.
    
    Args:
        url (str): URL of the issue page
        logger (logging.Logger): Logger instance
    
    Returns:
        dict: Dictionary containing description and labels, or None if failed
    """
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=10, verify=False)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'lxml')
        
        # Extract issue description
        description = ""
        desc_elements = soup.find_all(['p', 'div'], class_=re.compile('description|issue-desc|comment'))
        if desc_elements:
            description = " ".join([elem.get_text(strip=True) for elem in desc_elements[:3]])
        
        # Extract labels/tags
        labels = []
        label_elements = soup.find_all(['span', 'div'], class_=re.compile('label|tag|chip'))
        if label_elements:
            labels = [elem.get_text(strip=True) for elem in label_elements]
        
        return {
            'description': description[:500] if description else "No description available",
            'labels': ', '.join(labels) if labels else "No labels"
        }
        
    except requests.exceptions.Timeout:
        logger.warning(f"[WARNING] Timeout for URL: {url}")
        return None
    except requests.exceptions.RequestException as e:
        logger.warning(f"[WARNING] Request failed for URL: {url} - {str(e)}")
        return None
    except Exception as e:
        logger.warning(f"[WARNING] Scraping error for URL: {url} - {str(e)}")
        return None


def scrape_all_issues(df, url_column, logger):
    """
    Scrape all issue URLs in the dataframe using multithreading.
    
    Args:
        df (pd.DataFrame): Dataframe containing URLs
        url_column (str): Name of the column containing URLs
        logger (logging.Logger): Logger instance
    
    Returns:
        tuple: (DataFrame with scraped data, successful count, failed count)
    """
    logger.info("Starting web scraping (multithreaded)...")
    
    results = {}
    tasks = []
    
    # Prepare tasks
    for idx, row in df.iterrows():
        url = row[url_column]
        if pd.isna(url) or not isinstance(url, str):
            results[idx] = None
            continue
            
        if not url.startswith('http'):
            url = 'https://' + url
        
        tasks.append((idx, url))
    
    # Execute with ThreadPoolExecutor
    successful_scrapes = 0
    failed_scrapes = 0
    
    # Use max_workers=20 for IO-bound tasks
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        # Submit all tasks
        future_to_idx = {
            executor.submit(scrape_issue_page, url, logger): idx 
            for idx, url in tasks
        }
        
        # Process results as they complete
        for future in tqdm(concurrent.futures.as_completed(future_to_idx), total=len(future_to_idx), desc="Scraping issues"):
            idx = future_to_idx[future]
            try:
                data = future.result()
                results[idx] = data
                if data:
                    successful_scrapes += 1
                else:
                    failed_scrapes += 1
            except Exception as e:
                results[idx] = None
                failed_scrapes += 1

    # Map results back to dataframe preserving order
    descriptions = []
    labels_list = []
    
    for idx in df.index:
        data = results.get(idx)
        if data:
            descriptions.append(data['description'])
            labels_list.append(data['labels'])
        else:
            descriptions.append("Scraping failed")
            labels_list.append("N/A")
            if idx not in results: # Count as failed if not in results (should be covered above but safe)
                failed_scrapes += 1

    df['scraped_description'] = descriptions
    df['scraped_labels'] = labels_list
    
    logger.info(f"[SUCCESS] Scraping complete: {successful_scrapes} successful, {failed_scrapes} failed")
    
    return df, successful_scrapes, failed_scrapes


# ============================================================================
# SECTION 5: CATEGORIZATION & ANALYSIS
# ============================================================================

def detect_pixel_model(text):
    """
    Detect Pixel model (8, 9, 10) from text using regex.
    
    Args:
        text (str): Text to analyze
    
    Returns:
        str: Detected Pixel model or "Unknown"
    """
    if pd.isna(text):
        return "Unknown"
    
    text = str(text).lower()
    
    # Regex patterns for Pixel models (including Pixel 2, 4, 7)
    patterns = [
        (r'pixel\s*10', 'Pixel 10'),
        (r'pixel\s*9', 'Pixel 9'),
        (r'pixel\s*8', 'Pixel 8'),
        (r'pixel\s*7', 'Pixel 7'),
        (r'pixel\s*6', 'Pixel 6'),
        (r'pixel\s*5', 'Pixel 5'),
        (r'pixel\s*4', 'Pixel 4'),
        (r'pixel\s*3', 'Pixel 3'),
        (r'pixel\s*2', 'Pixel 2'),
    ]
    
    for pattern, model in patterns:
        if re.search(pattern, text):
            return model
    
    return "Unknown"


def estimate_severity(text):
    """
    Estimate issue severity based on keywords.
    
    Args:
        text (str): Text to analyze
    
    Returns:
        str: Severity level (High, Medium, Low)
    """
    if pd.isna(text):
        return "Low"
    
    text = str(text).lower()
    
    # High severity keywords
    high_keywords = ['crash', 'freeze', 'unresponsive', 'shutdown', 'boot loop', 
                     'bricked', 'not working', 'dead', 'failure', 'failing', 'stuck']
    
    # Medium severity keywords
    medium_keywords = ['slow', 'lag', 'glitch', 'delay', 'stuttering', 
                       'performance', 'battery drain', 'overheating']
    
    # Low severity keywords
    low_keywords = ['cosmetic', 'typo', 'minor', 'ui issue', 'display issue']
    
    # Check for high severity
    for keyword in high_keywords:
        if keyword in text:
            return "High"
    
    # Check for medium severity
    for keyword in medium_keywords:
        if keyword in text:
            return "Medium"
    
    # Check for low severity or default
    for keyword in low_keywords:
        if keyword in text:
            return "Low"
    
    return "Low"


def categorize_issue(text):
    """
    Categorize issue based on keywords. Supports multiple categories.
    
    Args:
        text (str): Text to analyze
    
    Returns:
        list: List of matching categories
    """
    if pd.isna(text):
        return ["Others"]
    
    text = str(text).lower()
    categories = []
    
    # Category keywords
    category_keywords = {
        'Pixel 8': ['pixel 8', 'pixel8'],
        'Pixel 9': ['pixel 9', 'pixel9'],
        'Pixel 10': ['pixel 10', 'pixel10'],
        'Pixel 7': ['pixel 7', 'pixel7'],
        'Pixel 6': ['pixel 6', 'pixel6'],
        'Pixel 5': ['pixel 5', 'pixel5'],
        'Pixel 4': ['pixel 4', 'pixel4'],
        'Pixel 3': ['pixel 3', 'pixel3'],
        'Pixel 2': ['pixel 2', 'pixel2'],
        'Network Issues': ['wifi', 'network', 'connectivity', 'signal', 'mobile data', 
                          '5g', '4g', 'bluetooth', 'internet', 'webrtc'],
        'Hardware Issues': ['screen', 'display', 'battery', 'camera', 'speaker', 
                           'microphone', 'charging', 'power', 'sensor', 'hardware'],
        'Installation Issues': ['install', 'update', 'upgrade', 'setup', 'boot', 
                               'flash', 'rom', 'firmware'],
        'UI/Graphics Issues': ['shadow', 'outline', 'transform', 'render', 'visual', 
                              'color', 'pixel', 'graphics'],
        'Notification Issues': ['notification', 'alerts', 'banner'],
        'Testing/Build Issues': ['test', 'builder', 'chromium', 'failing', 'failure']
    }
    
    # Check each category
    for category, keywords in category_keywords.items():
        for keyword in keywords:
            if keyword in text:
                categories.append(category)
                break
    
    # If no category matches, assign "Others"
    if not categories:
        categories.append("Others")
    
    return categories


def apply_categorization(df, logger):
    """
    Apply all categorization logic to the dataframe.
    
    Args:
        df (pd.DataFrame): Input dataframe
        logger (logging.Logger): Logger instance
    
    Returns:
        pd.DataFrame: Dataframe with categorization columns added
    """
    logger.info("Applying categorization logic...")
    
    # Combine all text fields for analysis
    text_columns = []
    for col in ['Title', 'title', 'description', 'Description', 'scraped_description', 'scraped_labels']:
        if col in df.columns:
            text_columns.append(df[col].fillna('').astype(str))
    
    # Concatenate text columns row-wise
    if text_columns:
        # Create a dataframe from the list of series
        text_df = pd.concat(text_columns, axis=1)
        # Join row-wise
        df['combined_text'] = text_df.apply(lambda row: ' '.join(row.values.astype(str)), axis=1)
    else:
        df['combined_text'] = ''
    
    # Apply categorization functions
    df['pixel_model'] = df['combined_text'].apply(detect_pixel_model)
    df['severity'] = df['combined_text'].apply(estimate_severity)
    df['categories'] = df['combined_text'].apply(categorize_issue)
    
    # Create a primary category column (first category in the list)
    df['primary_category'] = df['categories'].apply(lambda x: x[0] if x else "Others")
    
    # Create a multi-category string for export
    df['all_categories'] = df['categories'].apply(lambda x: ', '.join(x))
    
    logger.info("[SUCCESS] Categorization complete")
    
    return df


# ============================================================================
# SECTION 6: VISUALIZATION
# ============================================================================

def create_bar_chart(df, logger):
    """
    Create a bar chart showing issue count per category.
    
    Args:
        df (pd.DataFrame): Dataframe with categorized data
        logger (logging.Logger): Logger instance
    """
    logger.info("Creating bar chart...")
    
    category_counts = df['primary_category'].value_counts()
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x=category_counts.index, y=category_counts.values, palette='viridis')
    plt.title('Issue Count by Category', fontsize=16, fontweight='bold')
    plt.xlabel('Category', fontsize=12)
    plt.ylabel('Number of Issues', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    plt.savefig('output/charts/bar_chart.png', dpi=300)
    plt.close()
    
    logger.info("[SUCCESS] Bar chart saved: output/charts/bar_chart.png")


def create_pie_chart(df, logger):
    """
    Create a pie chart showing category distribution percentage.
    
    Args:
        df (pd.DataFrame): Dataframe with categorized data
        logger (logging.Logger): Logger instance
    """
    logger.info("Creating pie chart...")
    
    category_counts = df['primary_category'].value_counts()
    
    plt.figure(figsize=(10, 8))
    colors = sns.color_palette('Set3', len(category_counts))
    plt.pie(category_counts.values, labels=category_counts.index, autopct='%1.1f%%',
            startangle=90, colors=colors)
    plt.title('Category Distribution', fontsize=16, fontweight='bold')
    plt.axis('equal')
    plt.tight_layout()
    
    plt.savefig('output/charts/pie_chart.png', dpi=300)
    plt.close()
    
    logger.info("[SUCCESS] Pie chart saved: output/charts/pie_chart.png")


def create_summary_table(df, logger):
    """
    Create a summary table of category counts.
    
    Args:
        df (pd.DataFrame): Dataframe with categorized data
        logger (logging.Logger): Logger instance
    
    Returns:
        pd.DataFrame: Summary dataframe
    """
    logger.info("Creating summary table...")
    
    summary = df['primary_category'].value_counts().reset_index()
    summary.columns = ['Category', 'Count']
    summary['Percentage'] = (summary['Count'] / summary['Count'].sum() * 100).round(2)
    
    return summary


# ============================================================================
# SECTION 7: EXPORT FUNCTIONS
# ============================================================================

def export_results(df, summary_df, logger, stats):
    """
    Export all results including cleaned data, summary, and log statistics.
    
    Args:
        df (pd.DataFrame): Cleaned dataframe
        summary_df (pd.DataFrame): Summary dataframe
        logger (logging.Logger): Logger instance
        stats (dict): Statistics dictionary
    """
    logger.info("Exporting results...")
    
    # Export cleaned data
    df.to_csv('output/cleaned_data.csv', index=False, encoding='utf-8-sig')
    logger.info("[SUCCESS] Cleaned data exported: output/cleaned_data.csv")
    
    # Export summary
    summary_df.to_csv('output/summary.csv', index=False, encoding='utf-8-sig')
    logger.info("[SUCCESS] Summary exported: output/summary.csv")
    
    # Write statistics to log
    logger.info("=" * 70)
    logger.info("FINAL STATISTICS")
    logger.info("=" * 70)
    logger.info(f"Total issues processed: {stats['total_issues']}")
    logger.info(f"Successful scrapes: {stats['successful_scrapes']}")
    logger.info(f"Failed scrapes: {stats['failed_scrapes']}")
    logger.info(f"Duplicates removed: {stats['duplicates_removed']}")
    logger.info("=" * 70)
    logger.info("[SUCCESS] Processing complete!")


# ============================================================================
# SECTION 8: URL COLUMN DETECTION
# ============================================================================

def detect_url_column(df, logger):
    """
    Intelligently detect which column contains URLs.
    
    Args:
        df (pd.DataFrame): Input dataframe
        logger (logging.Logger): Logger instance
    
    Returns:
        str: Name of URL column or None if not found
    """
    # Method 1: Check for 'ID' column (from your CSV format)
    if 'ID' in df.columns:
        # Check if IDs are numeric (issue tracker IDs)
        sample = df['ID'].dropna().iloc[0] if len(df['ID'].dropna()) > 0 else None
        try:
            int_sample = int(float(sample))  # Fix floats with .0
            logger.info(f"[SUCCESS] Found ID column - will construct URLs from IDs")
            return 'ID'
        except:
            pass

    
    # Method 2: Check for common URL column names
    common_url_names = ['url', 'link', 'issue_url', 'issue_link', 'web_url', 
                        'page_url', 'tracker_url', 'issue link', 'issue url']
    
    for col in df.columns:
        if col.lower() in common_url_names:
            logger.info(f"[SUCCESS] URL column detected: '{col}'")
            return col
    
    # Method 3: Check if any column name contains 'url' or 'link'
    for col in df.columns:
        if 'url' in col.lower() or 'link' in col.lower():
            logger.info(f"[SUCCESS] URL column detected: '{col}'")
            return col
    
    # Method 4: Check column contents for URLs
    for col in df.columns:
        if df[col].dtype == 'object':
            sample_value = df[col].dropna().iloc[0] if len(df[col].dropna()) > 0 else None
            if sample_value and isinstance(sample_value, str):
                if sample_value.startswith('http://') or sample_value.startswith('https://'):
                    logger.info(f"[SUCCESS] URL column detected by content: '{col}'")
                    return col
    
    return None


def construct_urls_from_ids(df, id_column, logger):
    """
    Construct Google Issue Tracker URLs from issue IDs.
    
    Args:
        df (pd.DataFrame): Input dataframe
        id_column (str): Name of the ID column
        logger (logging.Logger): Logger instance
    
    Returns:
        pd.DataFrame: Dataframe with new 'url' column
    """
    base_url = "https://issuetracker.google.com/issues/"
    df['url'] = df[id_column].apply(lambda x: base_url + str(int(float(x))) if pd.notna(x) else None)
    logger.info(f"[SUCCESS] Constructed {len(df)} URLs from ID column")
    return df


# ============================================================================
# SECTION 9: MAIN EXECUTION
# ============================================================================

def main():
    """
    Main execution function that orchestrates the entire pipeline.
    """
    # Step 1: Check dependencies
    check_and_install_dependencies()
    
    # Step 2: Setup directories and logging
    setup_directories()
    logger = setup_logging()
    
    # Step 3: Load CSV
    print("\n" + "=" * 70)
    print("STEP 1: LOADING CSV DATA")
    print("=" * 70)
    
    input_file = 'input/issueTracker 2(in) (1).csv'
    df = load_csv(input_file, logger)
    
    if df is None:
        logger.error("[ERROR] Cannot proceed without input data. Exiting.")
        return
    
    if len(df) == 0:
        logger.error("[ERROR] CSV file is empty. Please check your input file.")
        return
    
    original_count = len(df)
    
    # Step 4: Clean duplicates
    print("\n" + "=" * 70)
    print("STEP 2: REMOVING DUPLICATES")
    print("=" * 70)
    
    df = remove_duplicates(df, logger)
    duplicates_removed = original_count - len(df)
    
    # Step 5: Detect URL column or construct URLs
    print("\n" + "=" * 70)
    print("STEP 3: DETECTING/CONSTRUCTING URLs")
    print("=" * 70)
    
    url_column = detect_url_column(df, logger)
    
    if url_column == 'ID':
        # Construct URLs from IDs
        df = construct_urls_from_ids(df, 'ID', logger)
        url_column = 'url'
    elif url_column is None:
        logger.error("[ERROR] No URL column or ID column found in CSV.")
        logger.error("[INFO] Available columns: " + ", ".join(df.columns))
        logger.error("[INFO] Please ensure your CSV has either:")
        logger.error("[INFO]   - A column with 'url' or 'link' in the name")
        logger.error("[INFO]   - An 'ID' column with issue tracker IDs")
        return
    
    # Step 6: Scrape issue pages
    print("\n" + "=" * 70)
    print("STEP 4: SCRAPING ISSUE PAGES")
    print("=" * 70)
    
    df, successful_scrapes, failed_scrapes = scrape_all_issues(df, url_column, logger)
    
    # Step 7: Categorize and analyze
    print("\n" + "=" * 70)
    print("STEP 5: CATEGORIZATION & ANALYSIS")
    print("=" * 70)
    
    df = apply_categorization(df, logger)
    
    # Step 8: Create visualizations
    print("\n" + "=" * 70)
    print("STEP 6: CREATING VISUALIZATIONS")
    print("=" * 70)
    
    create_bar_chart(df, logger)
    create_pie_chart(df, logger)
    summary_df = create_summary_table(df, logger)
    
    # Step 9: Export results
    print("\n" + "=" * 70)
    print("STEP 7: EXPORTING RESULTS")
    print("=" * 70)
    
    stats = {
        'total_issues': original_count,
        'successful_scrapes': successful_scrapes,
        'failed_scrapes': failed_scrapes,
        'duplicates_removed': duplicates_removed
    }
    
    export_results(df, summary_df, logger, stats)
    
    print("\n" + "=" * 70)
    print("[SUCCESS] ALL TASKS COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    print(f"\nCheck the 'output' folder for:")
    print("  - cleaned_data.csv")
    print("  - summary.csv")
    print("  - charts/bar_chart.png")
    print("  - charts/pie_chart.png")
    print("  - logs/run_log_[timestamp].txt")


if __name__ == "__main__":
    main()