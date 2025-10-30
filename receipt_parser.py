import os
import sqlite3
import json
from pathlib import Path
import google.generativeai as genai
from PIL import Image
import hashlib

class ReceiptParser:
    def __init__(self, api_key, db_path="receipts.db"):
        """Initialize the receipt parser with Gemini API and database."""
        genai.configure(api_key=api_key)
        
        # List available models and select appropriate one
        print("Checking available models...")
        available_models = []
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                available_models.append(m.name)
                print(f"  - {m.name}")
        
        # Try models in order of preference (Flash models have higher free tier limits)
        model_preferences = [
            'models/gemini-2.0-flash',
            'models/gemini-2.5-flash',
            'models/gemini-flash-latest',
            'models/gemini-2.0-flash-exp',
            'models/gemini-1.5-flash-latest',
            'models/gemini-1.5-flash',
            'models/gemini-1.5-pro-latest',
            'models/gemini-1.5-pro',
            'models/gemini-pro-vision'
        ]
        
        selected_model = None
        for model_name in model_preferences:
            if model_name in available_models:
                selected_model = model_name
                break
        
        if not selected_model and available_models:
            # Use first available model with vision capability
            selected_model = available_models[0]
        
        if selected_model:
            print(f"Using model: {selected_model}")
            self.model = genai.GenerativeModel(selected_model)
        else:
            raise Exception("No suitable model found. Please check your API key and permissions.")
        
        self.db_path = db_path
        self.setup_database()
    
    def setup_database(self):
        """Create database and tables if they don't exist."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS receipts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_path TEXT NOT NULL,
                image_hash TEXT UNIQUE NOT NULL,
                receipt_date DATE,
                processed_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS items (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                receipt_id INTEGER,
                item_name TEXT NOT NULL,
                price REAL NOT NULL,
                FOREIGN KEY (receipt_id) REFERENCES receipts (id)
            )
        ''')
        
        conn.commit()
        conn.close()
        print(f"Database initialized at {self.db_path}")
    
    def calculate_image_hash(self, image_path):
        """Calculate SHA256 hash of an image file to detect duplicates."""
        sha256_hash = hashlib.sha256()
        with open(image_path, "rb") as f:
            # Read file in chunks to handle large images
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    
    def is_duplicate(self, image_hash):
        """Check if an image with this hash has already been processed."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(
            "SELECT id, image_path FROM receipts WHERE image_hash = ?",
            (image_hash,)
        )
        result = cursor.fetchone()
        conn.close()
        
        return result  # Returns (id, image_path) if duplicate, None otherwise
    
    def extract_items_from_receipt(self, image_path):
        """Use Gemini to extract items, prices, and date from receipt image."""
        try:
            print("Loading image...")
            img = Image.open(image_path)
            
            prompt = """
            Analyze this receipt image and extract the following information:
            1. The date from the receipt (format: YYYY-MM-DD)
            2. All food/product items with their prices
            
            CRITICAL RULES FOR ITEM NAMES:
            - Extract ONLY the generic product category name (e.g., "Bread", "Milk", "Eggs", "Tomato")
            - Remove ALL brand names, company names, and product line names
            - Remove ALL weight/quantity indicators (kg, g, ml, l, pieces, pack, etc.)
            - Remove ALL descriptive adjectives (fresh, organic, sliced, etc.)
            - Remove ALL package types (tetrapak, bottle, can, etc.)
            - Use singular form for countable items (e.g., "Tomato" not "Tomatoes")
            - For produce sold by weight, use just the item name (e.g., "Carrot" not "Carrots by weight kg")
            
            ITEMS TO EXCLUDE (do not include these):
            - Discounts, promotions, or savings (e.g., "Lidl Plus discount", "Sale", "Promo")
            - Deposits or refunds (e.g., "Deposit 0.10", "Bottle deposit")
            - Bags, packaging (e.g., "Paper bag", "Plastic bag", "Green bags")
            - Non-food items like cellophane, gift wrap, brushes, tools
            - Tax lines, subtotals, totals
            - Store loyalty program items
            
            TRANSLATE to simple English product names:
            - "Piens" → "Milk"
            - "Maize" → "Bread"
            - "Olas" → "Eggs"
            - "Siers" → "Cheese"
            - "Tomāti" → "Tomato"
            - "Burkāni" → "Carrot"
            - "Jogurts" → "Yogurt"
            - "Vistas filejas" → "Chicken"
            
            EXAMPLES OF CORRECT EXTRACTION:
            Receipt shows: "Rimi Smart Eggs A/M No.3 10pcs" → Extract as: "Eggs"
            Receipt shows: "Milk Rimi pasteurized tetrapak 2% 1l" → Extract as: "Milk"
            Receipt shows: "Carrots by weight kg" → Extract as: "Carrot"
            Receipt shows: "Druva Spekavota Sweet and Sour Bread" → Extract as: "Bread"
            Receipt shows: "Nongshim Neoguri Noodle Soup" → Extract as: "Noodle Soup"
            Receipt shows: "Fresh pork sausages" → Extract as: "Sausage"
            Receipt shows: "Lidl Plus discount -2.34" → EXCLUDE (it's a discount)
            Receipt shows: "Paper bag 0.19" → EXCLUDE (it's packaging)
            Receipt shows: "Deposit 0.10" → EXCLUDE (it's a deposit)
            
            Return the data as a JSON object with this exact format:
            {
                "date": "2024-10-04",
                "items": [
                    {"item_name": "Bread", "price": 1.80},
                    {"item_name": "Milk", "price": 0.71},
                    {"item_name": "Eggs", "price": 1.86}
                ]
            }
            
            If you cannot find a date, use null for the date field.
            If you cannot read the receipt clearly, return empty items array.
            """
            
            print("Sending request to Gemini API...")
            response = self.model.generate_content([prompt, img])
            print("Received response from Gemini API")
            
            # Extract JSON from response
            text = response.text.strip()
            
            # Remove markdown code blocks if present
            if text.startswith("```json"):
                text = text[7:]
            if text.startswith("```"):
                text = text[3:]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()
            
            data = json.loads(text)
            return data
            
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON response: {e}")
            print(f"Response text: {response.text}")
            return {"date": None, "items": []}
        except Exception as e:
            print(f"Error processing image: {e}")
            return {"date": None, "items": []}
    
    def save_to_database(self, image_path, image_hash, receipt_date, items):
        """Save extracted items and date to the database."""
        if not items:
            print("No items to save.")
            return
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Insert receipt record with hash and date
            cursor.execute(
                "INSERT INTO receipts (image_path, image_hash, receipt_date) VALUES (?, ?, ?)",
                (str(image_path), image_hash, receipt_date)
            )
            receipt_id = cursor.lastrowid
            
            # Insert items
            for item in items:
                cursor.execute(
                    "INSERT INTO items (receipt_id, item_name, price) VALUES (?, ?, ?)",
                    (receipt_id, item['item_name'], item['price'])
                )
            
            conn.commit()
            print(f"Saved {len(items)} items to database (Receipt ID: {receipt_id})")
            
        except Exception as e:
            conn.rollback()
            print(f"Error saving to database: {e}")
        finally:
            conn.close()
    
    def process_receipt(self, image_path):
        """Process a receipt image and save results to database."""
        print(f"\nProcessing: {image_path}")
        
        # Calculate image hash to check for duplicates
        image_hash = self.calculate_image_hash(image_path)
        
        # Check if this receipt has already been processed
        duplicate = self.is_duplicate(image_hash)
        if duplicate:
            receipt_id, old_path = duplicate
            print(f"⚠️  DUPLICATE DETECTED - This receipt was already processed")
            print(f"   Original: {Path(old_path).name} (Receipt ID: {receipt_id})")
            print(f"   Skipping processing to avoid duplicate data.")
            return None
        
        data = self.extract_items_from_receipt(image_path)
        receipt_date = data.get('date')
        items = data.get('items', [])
        
        # Set to 'UNKNOWN' if date not found
        if not receipt_date:
            receipt_date = 'UNKNOWN'
            print("Receipt Date: UNKNOWN")
        else:
            print(f"Receipt Date: {receipt_date}")
        
        if items:
            print(f"Extracted {len(items)} items:")
            for item in items:
                print(f"  - {item['item_name']}: ${item['price']:.2f}")
            
            self.save_to_database(image_path, image_hash, receipt_date, items)
        else:
            print("No items extracted from receipt.")
        
        return data
    
    def process_receipts_folder(self, folder_path="receipts"):
        """Process all receipt images in a folder."""
        folder = Path(folder_path)
        
        if not folder.exists():
            print(f"Creating folder: {folder_path}")
            folder.mkdir(parents=True, exist_ok=True)
            print(f"Please add receipt images to the '{folder_path}' folder and run again.")
            return
        
        # Supported image formats
        image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}
        image_files = [f for f in folder.iterdir() 
                      if f.is_file() and f.suffix.lower() in image_extensions]
        
        if not image_files:
            print(f"No image files found in '{folder_path}' folder.")
            print(f"Supported formats: {', '.join(image_extensions)}")
            return
        
        print(f"\nFound {len(image_files)} receipt image(s) to process")
        print("="*60)
        
        processed_count = 0
        skipped_count = 0
        
        for idx, image_file in enumerate(image_files, 1):
            print(f"\n[{idx}/{len(image_files)}] Processing: {image_file.name}")
            print("-"*60)
            result = self.process_receipt(image_file)
            
            if result is None:
                skipped_count += 1
            else:
                processed_count += 1
        
        print("\n" + "="*60)
        print("Processing complete!")
        print(f"  ✓ New receipts processed: {processed_count}")
        print(f"  ⊘ Duplicates skipped: {skipped_count}")
        print("="*60)
    
    def get_all_items(self):
        """Retrieve all items from the database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT r.id, r.image_path, r.receipt_date, r.processed_date, i.item_name, i.price
            FROM receipts r
            JOIN items i ON r.id = i.receipt_id
            ORDER BY r.processed_date DESC, i.item_name
        ''')
        
        results = cursor.fetchall()
        conn.close()
        
        return results


def main():
    """Main function to demonstrate usage."""
    
    # Get API key from environment variable
    api_key = os.getenv("GEMINI_API_KEY")
    
    if not api_key:
        print("Error: Please set GEMINI_API_KEY environment variable")
        print("Example: export GEMINI_API_KEY='your-api-key-here'")
        return
    
    # Initialize parser
    parser = ReceiptParser(api_key)
    
    # Process all receipts in the 'receipts' folder
    parser.process_receipts_folder("receipts")
    
    # Display all items in database
    print("\n" + "="*60)
    print("ALL ITEMS IN DATABASE:")
    print("="*60)
    
    items = parser.get_all_items()
    
    if not items:
        print("No items found in database.")
        return
    
    current_receipt = None
    
    for receipt_id, image_path, receipt_date, processed_date, item_name, price in items:
        if receipt_id != current_receipt:
            date_str = receipt_date if receipt_date and receipt_date != 'UNKNOWN' else "UNKNOWN"
            print(f"\nReceipt #{receipt_id} - {date_str}")
            print(f"  File: {Path(image_path).name}")
            print(f"  Processed: {processed_date}")
            print(f"  Items:")
            current_receipt = receipt_id
        print(f"    • {item_name}: ${price:.2f}")


if __name__ == "__main__":
    main()