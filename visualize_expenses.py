import sqlite3
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
from pathlib import Path
import seaborn as sns

class ExpenseVisualizer:
    def __init__(self, db_path="receipts.db"):
        """Initialize the expense visualizer with database connection."""
        self.db_path = db_path
        if not Path(db_path).exists():
            raise FileNotFoundError(f"Database not found: {db_path}")
        
        # Set style for better-looking charts
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
    
    def load_data(self):
        """Load data from database into pandas DataFrames."""
        conn = sqlite3.connect(self.db_path)
        
        # Load receipts data
        self.receipts_df = pd.read_sql_query(
            "SELECT * FROM receipts", 
            conn
        )
        
        # Load items data with receipt information
        self.items_df = pd.read_sql_query("""
            SELECT 
                i.id,
                i.receipt_id,
                i.item_name,
                i.price,
                r.receipt_date,
                r.processed_date
            FROM items i
            JOIN receipts r ON i.receipt_id = r.id
        """, conn)
        
        conn.close()
        
        # Convert dates to datetime, handling UNKNOWN dates
        self.receipts_df['receipt_date'] = pd.to_datetime(
            self.receipts_df['receipt_date'], 
            errors='coerce'
        )
        self.items_df['receipt_date'] = pd.to_datetime(
            self.items_df['receipt_date'], 
            errors='coerce'
        )
        
        print(f"Loaded {len(self.receipts_df)} receipts and {len(self.items_df)} items")
    
    def plot_spending_by_receipt(self):
        """Bar chart: Total spending per receipt."""
        receipt_totals = self.items_df.groupby('receipt_id')['price'].sum().reset_index()
        receipt_totals = receipt_totals.merge(
            self.receipts_df[['id', 'receipt_date']], 
            left_on='receipt_id', 
            right_on='id'
        )
        receipt_totals = receipt_totals.sort_values('receipt_date')
        
        plt.figure(figsize=(12, 6))
        bars = plt.bar(
            range(len(receipt_totals)), 
            receipt_totals['price'],
            color='steelblue',
            edgecolor='black'
        )
        
        # Color bars with unknown dates differently
        for i, (idx, row) in enumerate(receipt_totals.iterrows()):
            if pd.isna(row['receipt_date']):
                bars[i].set_color('lightcoral')
        
        plt.xlabel('Receipt Number', fontsize=12, fontweight='bold')
        plt.ylabel('Total Amount ($)', fontsize=12, fontweight='bold')
        plt.title('Total Spending per Receipt', fontsize=14, fontweight='bold')
        plt.xticks(range(len(receipt_totals)), receipt_totals['receipt_id'])
        plt.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for i, v in enumerate(receipt_totals['price']):
            plt.text(i, v + 0.5, f'${v:.2f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig('spending_by_receipt.png', dpi=300, bbox_inches='tight')
        print("✓ Created: spending_by_receipt.png")
    
    def plot_top_items_by_frequency(self, top_n=15):
        """Bar chart: Most frequently purchased items."""
        item_frequency = self.items_df['item_name'].value_counts().head(top_n)
        
        plt.figure(figsize=(12, 8))
        bars = plt.barh(range(len(item_frequency)), item_frequency.values, color='mediumseagreen')
        plt.yticks(range(len(item_frequency)), item_frequency.index)
        plt.xlabel('Number of Purchases', fontsize=12, fontweight='bold')
        plt.ylabel('Item Name', fontsize=12, fontweight='bold')
        plt.title(f'Top {top_n} Most Frequently Purchased Items', fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()
        
        # Add value labels
        for i, v in enumerate(item_frequency.values):
            plt.text(v + 0.1, i, str(v), va='center', fontsize=9)
        
        plt.tight_layout()
        plt.savefig('top_items_frequency.png', dpi=300, bbox_inches='tight')
        print("✓ Created: top_items_frequency.png")
    
    def plot_top_items_by_spending(self, top_n=15):
        """Bar chart: Items with highest total spending."""
        item_spending = self.items_df.groupby('item_name')['price'].sum().sort_values(ascending=False).head(top_n)
        
        plt.figure(figsize=(12, 8))
        bars = plt.barh(range(len(item_spending)), item_spending.values, color='coral')
        plt.yticks(range(len(item_spending)), item_spending.index)
        plt.xlabel('Total Spending ($)', fontsize=12, fontweight='bold')
        plt.ylabel('Item Name', fontsize=12, fontweight='bold')
        plt.title(f'Top {top_n} Items by Total Spending', fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()
        
        # Add value labels
        for i, v in enumerate(item_spending.values):
            plt.text(v + 0.3, i, f'${v:.2f}', va='center', fontsize=9)
        
        plt.tight_layout()
        plt.savefig('top_items_spending.png', dpi=300, bbox_inches='tight')
        print("✓ Created: top_items_spending.png")
    
    def plot_spending_over_time(self):
        """Line chart: Spending trends over time (only for receipts with dates)."""
        # Filter out receipts with unknown dates
        dated_items = self.items_df[self.items_df['receipt_date'].notna()].copy()
        
        if len(dated_items) == 0:
            print("⚠ Skipped: spending_over_time.png (no receipts with dates)")
            return
        
        # Group by date and sum spending
        daily_spending = dated_items.groupby('receipt_date')['price'].sum().reset_index()
        daily_spending = daily_spending.sort_values('receipt_date')
        
        plt.figure(figsize=(14, 6))
        plt.plot(daily_spending['receipt_date'], daily_spending['price'], 
                marker='o', linewidth=2, markersize=8, color='darkblue')
        plt.xlabel('Date', fontsize=12, fontweight='bold')
        plt.ylabel('Total Spending ($)', fontsize=12, fontweight='bold')
        plt.title('Spending Over Time', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45, ha='right')
        
        # Add value labels
        for x, y in zip(daily_spending['receipt_date'], daily_spending['price']):
            plt.text(x, y + 1, f'${y:.2f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig('spending_over_time.png', dpi=300, bbox_inches='tight')
        print("✓ Created: spending_over_time.png")
    
    def plot_spending_by_month(self):
        """Bar chart: Monthly spending analysis."""
        dated_items = self.items_df[self.items_df['receipt_date'].notna()].copy()
        
        if len(dated_items) == 0:
            print("⚠ Skipped: spending_by_month.png (no receipts with dates)")
            return
        
        dated_items['month'] = dated_items['receipt_date'].dt.to_period('M')
        monthly_spending = dated_items.groupby('month')['price'].sum().reset_index()
        monthly_spending['month_str'] = monthly_spending['month'].astype(str)
        
        plt.figure(figsize=(14, 6))
        bars = plt.bar(range(len(monthly_spending)), monthly_spending['price'], 
                      color='teal', edgecolor='black')
        plt.xticks(range(len(monthly_spending)), monthly_spending['month_str'], rotation=45, ha='right')
        plt.xlabel('Month', fontsize=12, fontweight='bold')
        plt.ylabel('Total Spending ($)', fontsize=12, fontweight='bold')
        plt.title('Monthly Spending', fontsize=14, fontweight='bold')
        plt.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for i, v in enumerate(monthly_spending['price']):
            plt.text(i, v + 1, f'${v:.2f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig('spending_by_month.png', dpi=300, bbox_inches='tight')
        print("✓ Created: spending_by_month.png")
    
    def plot_price_distribution(self):
        """Histogram: Distribution of item prices."""
        plt.figure(figsize=(12, 6))
        plt.hist(self.items_df['price'], bins=30, color='purple', edgecolor='black', alpha=0.7)
        plt.xlabel('Price ($)', fontsize=12, fontweight='bold')
        plt.ylabel('Frequency', fontsize=12, fontweight='bold')
        plt.title('Distribution of Item Prices', fontsize=14, fontweight='bold')
        plt.grid(axis='y', alpha=0.3)
        
        # Add statistics
        mean_price = self.items_df['price'].mean()
        median_price = self.items_df['price'].median()
        plt.axvline(mean_price, color='red', linestyle='--', linewidth=2, label=f'Mean: ${mean_price:.2f}')
        plt.axvline(median_price, color='green', linestyle='--', linewidth=2, label=f'Median: ${median_price:.2f}')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('price_distribution.png', dpi=300, bbox_inches='tight')
        print("✓ Created: price_distribution.png")
    
    def plot_items_per_receipt(self):
        """Bar chart: Number of items per receipt."""
        items_per_receipt = self.items_df.groupby('receipt_id').size().reset_index(name='item_count')
        
        plt.figure(figsize=(12, 6))
        bars = plt.bar(range(len(items_per_receipt)), items_per_receipt['item_count'], 
                      color='orange', edgecolor='black')
        plt.xticks(range(len(items_per_receipt)), items_per_receipt['receipt_id'])
        plt.xlabel('Receipt Number', fontsize=12, fontweight='bold')
        plt.ylabel('Number of Items', fontsize=12, fontweight='bold')
        plt.title('Number of Items per Receipt', fontsize=14, fontweight='bold')
        plt.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for i, v in enumerate(items_per_receipt['item_count']):
            plt.text(i, v + 0.2, str(v), ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig('items_per_receipt.png', dpi=300, bbox_inches='tight')
        print("✓ Created: items_per_receipt.png")
    
    def generate_summary_statistics(self):
        """Generate and display summary statistics."""
        print("\n" + "="*60)
        print("EXPENSE SUMMARY STATISTICS")
        print("="*60)
        
        total_receipts = len(self.receipts_df)
        total_items = len(self.items_df)
        total_spending = self.items_df['price'].sum()
        avg_receipt_total = self.items_df.groupby('receipt_id')['price'].sum().mean()
        avg_item_price = self.items_df['price'].mean()
        
        print(f"\n📊 Overall Statistics:")
        print(f"  • Total Receipts: {total_receipts}")
        print(f"  • Total Items Purchased: {total_items}")
        print(f"  • Total Spending: ${total_spending:.2f}")
        print(f"  • Average per Receipt: ${avg_receipt_total:.2f}")
        print(f"  • Average Item Price: ${avg_item_price:.2f}")
        
        print(f"\n🏆 Top 5 Most Purchased Items:")
        top_items = self.items_df['item_name'].value_counts().head(5)
        for idx, (item, count) in enumerate(top_items.items(), 1):
            print(f"  {idx}. {item}: {count} times")
        
        print(f"\n💰 Top 5 Most Expensive Items:")
        top_expensive = self.items_df.nlargest(5, 'price')[['item_name', 'price']]
        for idx, (_, row) in enumerate(top_expensive.iterrows(), 1):
            print(f"  {idx}. {row['item_name']}: ${row['price']:.2f}")
        
        # Date range for receipts with dates
        dated_receipts = self.receipts_df[self.receipts_df['receipt_date'].notna()]
        if len(dated_receipts) > 0:
            min_date = dated_receipts['receipt_date'].min()
            max_date = dated_receipts['receipt_date'].max()
            print(f"\n📅 Date Range:")
            print(f"  • From: {min_date.strftime('%Y-%m-%d')}")
            print(f"  • To: {max_date.strftime('%Y-%m-%d')}")
        
        print("\n" + "="*60 + "\n")
    
    def generate_all_visualizations(self):
        """Generate all visualization charts."""
        print("\n" + "="*60)
        print("GENERATING EXPENSE VISUALIZATIONS")
        print("="*60 + "\n")
        
        self.load_data()
        self.generate_summary_statistics()
        
        print("Creating charts...")
        self.plot_spending_by_receipt()
        self.plot_top_items_by_frequency()
        self.plot_top_items_by_spending()
        self.plot_spending_over_time()
        self.plot_spending_by_month()
        self.plot_price_distribution()
        self.plot_items_per_receipt()
        
        print("\n" + "="*60)
        print("✅ ALL VISUALIZATIONS CREATED SUCCESSFULLY!")
        print("="*60)
        print("\nGenerated files:")
        print("  1. spending_by_receipt.png - Total spending per receipt")
        print("  2. top_items_frequency.png - Most frequently purchased items")
        print("  3. top_items_spending.png - Items with highest total spending")
        print("  4. spending_over_time.png - Spending trends over time")
        print("  5. spending_by_month.png - Monthly spending analysis")
        print("  6. price_distribution.png - Distribution of item prices")
        print("  7. items_per_receipt.png - Number of items per receipt")
        print()


def main():
    """Main function to run the visualizer."""
    try:
        visualizer = ExpenseVisualizer()
        visualizer.generate_all_visualizations()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run receipt_parser.py first to create the database.")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()