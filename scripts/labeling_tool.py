"""
Financial Document Labeling Tool

A simple command-line tool for manually labeling financial documents.
Run this script to label documents one by one and save labels to CSV.

Usage:
    python labeling_tool.py

The tool will:
1. Load unlabeled documents from data/raw/
2. Display document content (or summary)
3. Prompt for labels
4. Save labels to data/labeled/labels.csv
"""

import os
import csv
from datetime import datetime
from pathlib import Path


# Configuration
RAW_DATA_DIR = Path("data/raw")
LABELS_FILE = Path("data/labeled/labels.csv")
COLLECTION_LOG = Path("data/collection_log.csv")

# Valid label options
SENTIMENT_OPTIONS = ["positive", "negative", "neutral"]
RISK_OPTIONS = ["low", "medium", "high"]
OUTLOOK_OPTIONS = ["bullish", "bearish", "neutral"]


def load_collection_log():
    """Load the collection log to get document metadata."""
    documents = []
    if COLLECTION_LOG.exists():
        with open(COLLECTION_LOG, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                documents.append(row)
    return documents


def load_existing_labels():
    """Load already labeled document IDs."""
    labeled_ids = set()
    if LABELS_FILE.exists():
        with open(LABELS_FILE, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get('document_id'):
                    labeled_ids.add(row['document_id'])
    return labeled_ids


def read_document(file_path):
    """Read document content from file."""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        return content
    except Exception as e:
        print(f"Error reading file: {e}")
        return None


def display_document_preview(content, max_chars=2000):
    """Display a preview of the document content."""
    print("\n" + "="*80)
    print("DOCUMENT PREVIEW")
    print("="*80)
    
    if len(content) > max_chars:
        print(content[:max_chars])
        print(f"\n... [Truncated - {len(content)} total characters] ...")
    else:
        print(content)
    
    print("="*80 + "\n")


def get_valid_input(prompt, valid_options):
    """Get validated input from user."""
    while True:
        print(f"{prompt}")
        print(f"Options: {', '.join(valid_options)}")
        user_input = input("Your choice: ").strip().lower()
        
        if user_input in valid_options:
            return user_input
        elif user_input == 'skip':
            return 'skip'
        elif user_input == 'quit':
            return 'quit'
        else:
            print(f"Invalid input. Please enter one of: {', '.join(valid_options)}")
            print("Or type 'skip' to skip this document, 'quit' to exit.\n")


def save_label(document_info, sentiment, risk_level, market_outlook, labeler_name):
    """Append label to the labels CSV file."""
    
    fieldnames = [
        'document_id', 'source', 'company_name', 'ticker', 'sector',
        'document_type', 'year', 'quarter', 'collection_date', 'file_path',
        'sentiment', 'risk_level', 'market_outlook', 'labeled_by', 'label_date', 'notes'
    ]
    
    row = {
        'document_id': document_info.get('document_id', ''),
        'source': document_info.get('source', ''),
        'company_name': document_info.get('company_name', ''),
        'ticker': document_info.get('ticker', ''),
        'sector': document_info.get('sector', ''),
        'document_type': document_info.get('document_type', ''),
        'year': document_info.get('year', ''),
        'quarter': document_info.get('quarter', ''),
        'collection_date': document_info.get('collection_date', ''),
        'file_path': document_info.get('file_path', ''),
        'sentiment': sentiment,
        'risk_level': risk_level,
        'market_outlook': market_outlook,
        'labeled_by': labeler_name,
        'label_date': datetime.now().strftime('%Y-%m-%d'),
        'notes': document_info.get('notes', '')
    }
    
    file_exists = LABELS_FILE.exists() and LABELS_FILE.stat().st_size > 0
    
    with open(LABELS_FILE, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
    
    print(f"Label saved for document: {document_info.get('document_id', 'unknown')}")


def label_single_document(file_path, labeler_name):
    """Label a single document that may not be in the collection log."""
    
    content = read_document(file_path)
    if content is None:
        return False
    
    display_document_preview(content)
    
    # Get document info from filename
    filename = Path(file_path).stem
    
    print(f"\nLabeling: {filename}")
    print("-" * 40)
    
    # Get labels
    sentiment = get_valid_input("\n1. What is the overall SENTIMENT?", SENTIMENT_OPTIONS)
    if sentiment == 'quit':
        return False
    if sentiment == 'skip':
        print("Skipping this document.")
        return True
    
    risk_level = get_valid_input("\n2. What is the RISK LEVEL?", RISK_OPTIONS)
    if risk_level == 'quit':
        return False
    if risk_level == 'skip':
        print("Skipping this document.")
        return True
    
    market_outlook = get_valid_input("\n3. What is the MARKET OUTLOOK?", OUTLOOK_OPTIONS)
    if market_outlook == 'quit':
        return False
    if market_outlook == 'skip':
        print("Skipping this document.")
        return True
    
    # Create document info
    document_info = {
        'document_id': filename,
        'file_path': str(file_path),
        'source': 'manual',
        'company_name': '',
        'ticker': '',
        'sector': '',
        'document_type': '',
        'year': '',
        'quarter': '',
        'collection_date': datetime.now().strftime('%Y-%m-%d'),
        'notes': ''
    }
    
    save_label(document_info, sentiment, risk_level, market_outlook, labeler_name)
    return True


def main():
    """Main labeling loop."""
    
    print("\n" + "="*80)
    print("FINANCIAL DOCUMENT LABELING TOOL")
    print("="*80)
    print("\nThis tool helps you label financial documents for the deep learning project.")
    print("You will be asked to classify each document by:")
    print("  - Sentiment (positive, negative, neutral)")
    print("  - Risk Level (low, medium, high)")
    print("  - Market Outlook (bullish, bearish, neutral)")
    print("\nType 'skip' to skip a document, 'quit' to exit at any time.")
    print("="*80)
    
    # Get labeler name
    labeler_name = input("\nEnter your name (for tracking): ").strip()
    if not labeler_name:
        labeler_name = "anonymous"
    
    # Load existing data
    collection_log = load_collection_log()
    labeled_ids = load_existing_labels()
    
    # Find unlabeled documents
    unlabeled_docs = [doc for doc in collection_log if doc.get('document_id') not in labeled_ids]
    
    if not unlabeled_docs:
        # Check raw directory for any files
        if RAW_DATA_DIR.exists():
            raw_files = list(RAW_DATA_DIR.glob('*.txt')) + list(RAW_DATA_DIR.glob('*.html'))
            if raw_files:
                print(f"\nFound {len(raw_files)} files in {RAW_DATA_DIR}")
                print("These files are not in the collection log. Would you like to label them anyway?")
                response = input("Enter 'yes' to proceed: ").strip().lower()
                
                if response == 'yes':
                    for file_path in raw_files:
                        if file_path.stem not in labeled_ids:
                            success = label_single_document(file_path, labeler_name)
                            if not success:
                                break
                    print("\nLabeling session complete.")
                    return
        
        print("\nNo unlabeled documents found.")
        print("Please add documents to data/raw/ and update data/collection_log.csv")
        return
    
    print(f"\nFound {len(unlabeled_docs)} unlabeled documents.")
    print("Starting labeling session...\n")
    
    # Label each document
    labeled_count = 0
    for i, doc in enumerate(unlabeled_docs):
        print(f"\n[{i+1}/{len(unlabeled_docs)}] Document: {doc.get('document_id', 'unknown')}")
        print(f"Company: {doc.get('company_name', 'N/A')} | Type: {doc.get('document_type', 'N/A')}")
        
        file_path = doc.get('file_path', '')
        if not file_path or not Path(file_path).exists():
            print(f"Warning: File not found at {file_path}. Skipping.")
            continue
        
        content = read_document(file_path)
        if content is None:
            continue
        
        display_document_preview(content)
        
        # Get labels
        sentiment = get_valid_input("1. What is the overall SENTIMENT?", SENTIMENT_OPTIONS)
        if sentiment == 'quit':
            break
        if sentiment == 'skip':
            print("Skipping this document.")
            continue
        
        risk_level = get_valid_input("2. What is the RISK LEVEL?", RISK_OPTIONS)
        if risk_level == 'quit':
            break
        if risk_level == 'skip':
            print("Skipping this document.")
            continue
        
        market_outlook = get_valid_input("3. What is the MARKET OUTLOOK?", OUTLOOK_OPTIONS)
        if market_outlook == 'quit':
            break
        if market_outlook == 'skip':
            print("Skipping this document.")
            continue
        
        save_label(doc, sentiment, risk_level, market_outlook, labeler_name)
        labeled_count += 1
    
    print(f"\n{'='*80}")
    print(f"Labeling session complete. Documents labeled: {labeled_count}")
    print(f"Labels saved to: {LABELS_FILE}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
