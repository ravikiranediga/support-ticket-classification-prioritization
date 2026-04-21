"""
Training Script
Trains both ticket type and priority classification models
"""

import sys
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd

from preprocessing import combine_ticket
from model import TicketClassifier
from config import RAW_DATA_PATH, TARGET_TYPE, TARGET_PRIORITY


def load_and_prepare_data(csv_path):
    """Load CSV and add preprocessed text column"""
    print(f"Loading data from: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"Total tickets: {len(df)}")

    # Create combined text feature
    print("Preprocessing ticket text...")
    df['combined_text'] = df.apply(
        lambda row: combine_ticket(row['Ticket Subject'], row['Ticket Description']),
        axis=1
    )

    return df


def train_models():
    """Train both ticket type and priority models"""
    print("\n" + "="*70)
    print("SUPPORT TICKET CLASSIFICATION SYSTEM")
    print("="*70)

    # Load data
    df = load_and_prepare_data(RAW_DATA_PATH)

    results = {}

    # ========================
    # 1. Train Ticket Type Model
    # ========================
    print(f"\n{'#'*70}")
    print(f"# MODEL 1: {TARGET_TYPE}")
    print(f"{'#'*70}")

    type_clf = TicketClassifier(target_name=TARGET_TYPE)
    results['ticket_type'] = type_clf.train(df, text_column='combined_text')

    # ========================
    # 2. Train Priority Model
    # ========================
    print(f"\n{'#'*70}")
    print(f"# MODEL 2: {TARGET_PRIORITY}")
    print(f"{'#'*70}")

    priority_clf = TicketClassifier(target_name=TARGET_PRIORITY)
    results['ticket_priority'] = priority_clf.train(df, text_column='combined_text')

    # Summary
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    print(f"\nTicket Type Accuracy:  {results['ticket_type']['accuracy']:.4f}")
    print(f"Priority Accuracy:     {results['ticket_priority']['accuracy']:.4f}")
    print("\nModel files saved in: ./models/")
    print("Charts saved in: ./reports/")

    return results


if __name__ == "__main__":
    train_models()
