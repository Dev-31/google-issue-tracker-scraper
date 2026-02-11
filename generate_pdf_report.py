"""
Generate PDF Report of Code Explanation
=======================================
This script generates a PDF document explaining the codebase of the 
Google Issue Tracker Scraper project.
"""

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from datetime import datetime

def create_code_explanation_pdf(filename="Project_Code_Explanation.pdf"):
    doc = SimpleDocTemplate(filename, pagesize=letter)
    styles = getSampleStyleSheet()
    
    # Custom Styles
    title_style = styles['Title']
    heading_style = styles['Heading1']
    subheading_style = styles['Heading2']
    normal_style = styles['Normal']
    code_style = ParagraphStyle(
        'Code',
        parent=styles['Code'],
        fontSize=8,
        leading=10,
        backColor=colors.lightgrey,
        borderPadding=5,
        spaceBefore=5,
        spaceAfter=5
    )

    story = []

    # Title Page
    story.append(Paragraph("Google Issue Tracker Scraper", title_style))
    story.append(Paragraph("Codebase Explanation & Technical Documentation", styles['Heading2']))
    story.append(Spacer(1, 0.5*inch))
    story.append(Paragraph(f"Date: {datetime.now().strftime('%Y-%m-%d')}", normal_style))
    story.append(Spacer(1, 2*inch))
    story.append(Paragraph("This document provides a detailed technical explanation of the Python script designed to scrape, analyze, and visualize data from the Google Issue Tracker.", normal_style))
    story.append(PageBreak())

    # Table of Contents
    story.append(Paragraph("Table of Contents", heading_style))
    story.append(Paragraph("1. Project Overview", normal_style))
    story.append(Paragraph("2. Key Libraries & Dependencies", normal_style))
    story.append(Paragraph("3. Core Architecture", normal_style))
    story.append(Paragraph("4. Detailed Code Breakdown", normal_style))
    story.append(Paragraph("   - Dependency Management", normal_style))
    story.append(Paragraph("   - Setup & Logging", normal_style))
    story.append(Paragraph("   - Data Loading & Cleaning", normal_style))
    story.append(Paragraph("   - Multithreaded Web Scraping", normal_style))
    story.append(Paragraph("   - Categorization Logic", normal_style))
    story.append(Paragraph("   - Visualization & Reporting", normal_style))
    story.append(PageBreak())

    # 1. Project Overview
    story.append(Paragraph("1. Project Overview", heading_style))
    story.append(Paragraph("""
    The Google Issue Tracker Scraper is a robust Python application that automates the extraction 
    of issue details from Google's public issue tracker. It takes a CSV file containing issue IDs or URLs 
    as input, scrapes detailed descriptions and labels for each issue, categorizes them based on 
    keywords (e.g., Pixel models, bug types), and generates actionable insights through charts and summary reports.
    """, normal_style))
    story.append(Spacer(1, 0.2*inch))

    # 2. Key Libraries
    story.append(Paragraph("2. Key Libraries & Dependencies", heading_style))
    data = [
        ["Library", "Purpose"],
        ["pandas", "Data manipulation and analysis (DataFrames)."],
        ["requests", "Making HTTP requests to fetch web pages."],
        ["BeautifulSoup", "Parsing HTML content to extract text."],
        ["concurrent.futures", "Handling multithreading for fast parallel scraping."],
        ["matplotlib / seaborn", "Generating bar charts and pie charts."],
        ["tqdm", "Displaying progress bars during long operations."]
    ]
    t = Table(data, colWidths=[2*inch, 4*inch])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(t)
    story.append(Spacer(1, 0.2*inch))

    # 3. Core Architecture
    story.append(Paragraph("3. Core Architecture", heading_style))
    story.append(Paragraph("""
    The application follows a linear pipeline architecture with distinct stages:
    Input -> Cleaning -> Scraping (Parallel) -> Analysis -> Visualization -> Output.
    """, normal_style))
    story.append(Spacer(1, 0.2*inch))

    # 4. Detailed Code Breakdown
    story.append(Paragraph("4. Detailed Code Breakdown", heading_style))

    # Dependency Management
    story.append(Paragraph("A. Dependency Management", subheading_style))
    story.append(Paragraph("""
    The script includes a self-check mechanism (`check_and_install_dependencies`) that runs at startup. 
    It verifies if all required packages are installed. If any are missing, it attempts to install them 
    automatically using `pip` via `subprocess`.
    """, normal_style))

    # Setup & Logging
    story.append(Paragraph("B. Setup & Logging", subheading_style))
    story.append(Paragraph("""
    To ensure robustness and traceability:
    <br/>- <b>Directories:</b> Uses `pathlib` to create input/output folders automatically.
    <br/>- <b>Logging:</b> Configures a dual-output logger (Console + File) that handles UTF-8 encoding, 
    which is critical for Windows systems to avoid emoji/character errors.
    """, normal_style))

    # Data Loading
    story.append(Paragraph("C. Data Loading & Cleaning", subheading_style))
    story.append(Paragraph("""
    The `load_csv` function handles file I/O. It attempts multiple file encodings (utf-8, latin-1, cp1252) 
    to prevent crashes when reading user-provided CSVs.
    <br/><br/>
    The `remove_duplicates` function intelligently identifies unique issues based on 'ID' or 'Title' columns 
    to ensure the analysis is accurate and efficient.
    """, normal_style))

    # Multithreading
    story.append(Paragraph("D. Multithreaded Web Scraping (The Core)", subheading_style))
    story.append(Paragraph("""
    This is the most critical part of the application for performance.
    <br/>- <b>Problem:</b> Scraping 1000 pages sequentially is too slow.
    <br/>- <b>Solution:</b> The `scrape_all_issues` function uses `concurrent.futures.ThreadPoolExecutor`.
    <br/>- <b>Implementation:</b> It spins up 20 worker threads that fetch pages in parallel. 
    It uses `urllib3` to suppress SSL warnings (needed for some corporate/proxy environments) and `requests` 
    to fetch the HTML. `BeautifulSoup` extracts the 'description' and 'labels' from the raw HTML.
    """, normal_style))
    story.append(Paragraph("""
    <font face="Courier">
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:<br/>
        # Submit tasks...<br/>
        # Process results as they complete...
    </font>
    """, code_style))

    # Categorization
    story.append(Paragraph("E. Categorization Logic", subheading_style))
    story.append(Paragraph("""
    Once text is scraped, the `apply_categorization` function runs.
    <br/>- <b>Pixel Model Detection:</b> Regex patterns (e.g., `pixel\s*9`) find specific device models.
    <br/>- <b>Severity Analysis:</b> Keywords like 'crash', 'dead', 'freeze' trigger 'High' severity.
    <br/>- <b>Topic Categorization:</b> A keyword dictionary maps terms to categories like 'UI/Graphics Issues', 
    'Network Issues', etc.
    """, normal_style))

    # Visualization
    story.append(Paragraph("F. Visualization & Reporting", subheading_style))
    story.append(Paragraph("""
    Finally, `matplotlib` and `seaborn` generate charts:
    <br/>- <b>Bar Chart:</b> Visualizes the count of issues per category.
    <br/>- <b>Pie Chart:</b> Shows the percentage distribution.
    <br/><br/>
    The `export_results` function saves the cleaned dataset (`cleaned_data.csv`) and a summary table (`summary.csv`).
    """, normal_style))

    # Conclusion
    story.append(Paragraph("Conclusion", heading_style))
    story.append(Paragraph("""
    This project demonstrates a complete data engineering pipeline: ingesting raw data, enriching it via 
    external sources (web scraping), transforming it (cleaning/categorizing), and loading it into 
    consumable formats (CSV reports and visualizations).
    """, normal_style))

    # Build PDF
    doc.build(story)
    print(f"PDF generated: {filename}")

if __name__ == "__main__":
    create_code_explanation_pdf()
