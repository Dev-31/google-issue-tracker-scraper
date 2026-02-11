"""
Generate Final Project Report PDF
==================================
This script generates a comprehensive project report PDF covering:
1. Requirements verification (Matched vs. Implemented)
2. HOW-TO-RUN guide
3. Code explanation and Problem/Solution summary
"""

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, ListFlowable, ListItem
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from datetime import datetime

def create_final_report_pdf(filename="Project_Implementation_Report.pdf"):
    doc = SimpleDocTemplate(filename, pagesize=letter)
    styles = getSampleStyleSheet()
    
    # Custom Styles
    title_style = styles['Title']
    heading1 = styles['Heading1']
    heading2 = styles['Heading2']
    normal_style = styles['Normal']
    code_style = ParagraphStyle(
        'Code',
        parent=styles['Code'],
        fontSize=8,
        leading=10,
        backColor=colors.lightgrey,
        borderPadding=5
    )

    story = []

    # --- Title Page ---
    story.append(Paragraph("Project Implementation Report", title_style))
    story.append(Paragraph("Google Issue Tracker AI Triage", styles['Heading2']))
    story.append(Spacer(1, 0.5*inch))
    story.append(Paragraph(f"Date: {datetime.now().strftime('%Y-%m-%d')}", normal_style))
    story.append(Spacer(1, 1*inch))
    
    story.append(Paragraph("<b>Table of Contents:</b>", heading2))
    story.append(Paragraph("1. Requirements Analysis (Gap Analysis)", normal_style))
    story.append(Paragraph("2. How to Run This Project", normal_style))
    story.append(Paragraph("3. Technical Implementation Details", normal_style))
    story.append(Paragraph("4. Challenges Faced & Solutions", normal_style))
    story.append(PageBreak())

    # --- 1. Requirements Analysis ---
    story.append(Paragraph("1. Requirements Analysis", heading1))
    story.append(Paragraph("A straightforward comparison of your requirements versus what is currently implemented.", normal_style))
    story.append(Spacer(1, 0.2*inch))

    # Requirement Implementation Table
    data = [
        ["Requirement Goal", "Status", "Implementation Notes"],
        ["<b>Step 1: Data Collection</b>", "PARTIAL", "We extract ID, Title, Description, but not full 'Labels', 'Creation Date' directly from the CSV unless scraping succeeds. We use scraping to get descriptions."],
        ["- Read Issue Tracker ID/Title", "DONE", "Reads from input CSV."],
        ["- Read Status/Labels/Dates", "PARTIAL", "Dependent on CSV columns. Scraper fetches descriptions."],
        ["<b>Step 2: Category 1 Classification</b>", "DONE", "Logic implemented in `detect_pixel_model` and `categorize_issue`."],
        ["- Primary Categories (Pixel X, Network)", "DONE", "Basic keyword matching is implemented."],
        ["- AI/Semantic Understanding", "BASIC", "Currently uses Keyword/Regex matching, not LLM-based understanding yet."],
        ["<b>Step 3: Visualization</b>", "DONE", "Bar charts and Summary tables are generated automatically."],
        ["- Count & Visualize", "DONE", "Matplotlib/Seaborn charts generated."]
    ]
    
    t = Table(data, colWidths=[2.5*inch, 1*inch, 3*inch])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.grey),
        ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('GRID', (0,0), (-1,-1), 1, colors.black),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.whitesmoke])
    ]))
    story.append(t)
    story.append(Spacer(1, 0.2*inch))
    story.append(Paragraph("<b>Summary:</b> We have a functional MVP. It successfully reads, scrapes, categorizes using logic rules, and visualizes the data. To truly meet the 'AI' requirement (semantic understanding), we would need to integrate an LLM API (like GPT/Gemini) instead of just Keyword matching.", normal_style))
    story.append(PageBreak())

    # --- 2. How to Run ---
    story.append(Paragraph("2. How to Run This Project", heading1))
    story.append(Paragraph("Follow these steps to execute the project on your local machine.", normal_style))

    # Prerequisites
    story.append(Paragraph("<b>Prerequisites:</b>", heading2))
    story.append(ListFlowable([
        ListItem(Paragraph("Python 3.8 or higher installed.", normal_style)),
        ListItem(Paragraph("An active internet connection.", normal_style)),
        ListItem(Paragraph("The input CSV file placed in `input/` folder.", normal_style))
    ], bulletType='bullet'))

    # Installation
    story.append(Paragraph("<b>Step 1: Install Dependencies</b>", heading2))
    story.append(Paragraph("Open your terminal in the project folder and run:", normal_style))
    story.append(Paragraph("pip install -r requirements.txt", code_style))

    # Execution
    story.append(Paragraph("<b>Step 2: Run the Scraper</b>", heading2))
    story.append(Paragraph("Execute the main script:", normal_style))
    story.append(Paragraph("python main.py", code_style))
    story.append(Paragraph("<i>Note: The script will automatically install missing dependencies if Step 1 was skipped.</i>", normal_style))

    # Output
    story.append(Paragraph("<b>Step 3: Check Results</b>", heading2))
    story.append(Paragraph("Navigate to the `output/` directory to see:", normal_style))
    story.append(ListFlowable([
        ListItem(Paragraph("<b>cleaned_data.csv:</b> The fully processed dataset.", normal_style)),
        ListItem(Paragraph("<b>summary.csv:</b> Classification counts.", normal_style)),
        ListItem(Paragraph("<b>charts/</b>: Visual graphs of the data.", normal_style))
    ], bulletType='bullet'))
    story.append(PageBreak())

    # --- 3. Technical Explanation ---
    story.append(Paragraph("3. Technical Implementation Details", heading1))
    story.append(Paragraph("The solution is a Python-based pipeline designed for speed and reliability.", normal_style))
    
    story.append(Paragraph("<b>Core Logic:</b>", heading2))
    story.append(ListFlowable([
        ListItem(Paragraph("<b>Data Ingestion:</b> Reads CSV files using Pandas, handling various encoding formats (utf-8, latin-1) automatically.", normal_style)),
        ListItem(Paragraph("<b>Multithreaded Scraping:</b> Uses `concurrent.futures` to scrape 20 pages in parallel. This was crucial for performance. It extracts descriptions from HTML using BeautifulSoup.", normal_style)),
        ListItem(Paragraph("<b>Categorization Engine:</b> A rule-based classifier assigns categories. <br/>- <i>e.g., If description contains 'Pixel 9', assign 'Pixel 9 Series'.</i><br/>- <i>e.g., If description contains 'wifi', assign 'Network Issues'.</i>", normal_style))
    ], bulletType='bullet'))
    story.append(Spacer(1, 0.2*inch))

    # --- 4. Challenges & Solutions ---
    story.append(Paragraph("4. Challenges Faced & Solutions", heading1))
    
    # Challenge 1
    story.append(Paragraph("<b>Challenge 1: SSL Certificate Verification Errors</b>", heading2))
    story.append(Paragraph("<b>Problem:</b> The script initially failed with `[SSL: CERTIFICATE_VERIFY_FAILED]`. This often happens in corporate network environments with proxies.", normal_style))
    story.append(Paragraph("<b>Solution:</b> We modified the `requests.get()` call to include `verify=False` and suppressed the resulting warnings using `urllib3`. This allows the scraper to function correctly in your environment.", normal_style))

    # Challenge 2
    story.append(Paragraph("<b>Challenge 2: Performance (Slow Scraping)</b>", heading2))
    story.append(Paragraph("<b>Problem:</b> Scraping ~800 issues one-by-one with Selenium or sequential logic would take 20-30 minutes.", normal_style))
    story.append(Paragraph("<b>Solution:</b> We switched to a **multithreaded** approach using Python's `ThreadPoolExecutor`. This reduced the execution time to under 1 minute by fetching 20 pages simultaneously.", normal_style))

    # Challenge 3
    story.append(Paragraph("<b>Challenge 3: Robustness</b>", heading2))
    story.append(Paragraph("<b>Problem:</b> The initial Selenium attempt was unstable and timed out.", normal_style))
    story.append(Paragraph("<b>Solution:</b> We reverted to `requests` + `BeautifulSoup` which is lighter and faster for static content. We kept the code modular so scraping logic is separate from analysis logic.", normal_style))

    doc.build(story)
    print(f"PDF generated: {filename}")

if __name__ == "__main__":
    create_final_report_pdf()
