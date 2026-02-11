# Google Issue Tracker Scraper & Analyzer

A comprehensive tool designed to scrape, clean, categorize, analyze, and visualize issues from the Google Issue Tracker. This project takes a CSV input containing issue IDs or URLs, enriches it with scraped data, and provides insightful visualizations and summaries.

## Features

*   **Automated Dependency Management**: Checks for required Python packages and installs them automatically if missing.
*   **Robust Data Loading**: Handles various CSV encodings (UTF-8, Latin-1, CP1252, etc.) to ensure data is loaded correctly.
*   **Smart Duplicate Removal**: Identifies and removes duplicate issues based on ID, Title, or Description.
*   **URL Detection**: Automatically detects URL columns or constructs valid Google Issue Tracker URLs from plain IDs.
*   **Dual Scraping Modes**:
    *   **Standard Mode (`main.py`)**: Uses `requests` and `BeautifulSoup` for fast, multithreaded scraping.
    *   **Selenium Mode (`selenium_main.py`)**: Uses a headless Chrome browser for handling dynamic content or bypassing potential blocks.
*   **Intelligent Categorization**:
    *   **Pixel Model Detection**: Identifies Pixel device models (e.g., Pixel 6, Pixel 7, Pixel 8) mentioned in the text.
    *   **Severity Estimation**: Estimates issue severity (High, Medium, Low) based on keyword analysis.
    *   **Issue Categorization**: Classifies issues into categories like Network, Hardware, UI/Graphics, Installation, etc.
*   **Visualization & Reporting**:
    *   Generates distribution Bar Charts and Pie Charts.
    *   Exports cleaned and enriched data to CSV.
    *   Creates summary statistics tables.
*   **Logging**: detailed logs preserved in `output/logs` for troubleshooting and auditing.

## Prerequisites

*   Python 3.x
*   Google Chrome (required for Selenium mode)

## Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/Dev-31/google-issue-tracker-scraper.git
    cd google-issue-tracker-scraper
    ```

2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
    *Note: The script also includes a self-check and will attempt to install missing dependencies on the first run.*

## Usage

1.  **Prepare Input**:
    *   Place your input CSV file in the `input/` directory.
    *   Ensure the CSV has a column with Issue IDs (e.g., `ID`) or URLs (e.g., `url`, `link`).

2.  **Run the Scraper**:

    **Option A: Fast Mode (Requests)**
    Use this for bulk scraping when the content is static.
    ```bash
    python main.py
    ```

    **Option B: Browser Mode (Selenium)**
    Use this if the standard mode fails to retrieve content or if the page requires JavaScript rendering.
    ```bash
    python selenium_main.py
    ```

## Output

After execution, check the `output/` directory for results:

*   `output/cleaned_data.csv`: The fully processed dataset with scraped descriptions, labels, and categorization tags.
*   `output/summary.csv`: A high-level summary of issue counts by category.
*   `output/charts/`: Contains generated visualizations (Bar Chart, Pie Chart).
*   `output/logs/`: Detailed execution logs.

## Project Structure

```
google-issue-tracker-scraper/
├── input/                  # Place your input CSV files here
├── output/                 # Generated results (CSV, Arrays, Logs)
│   ├── charts/             # Generated graphs
│   └── logs/               # Execution logs
├── main.py                 # Core script (Requests + Multithreading)
├── selenium_main.py        # Alternative script (Selenium WebDriver)
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

## License

[MIT License](LICENSE)
