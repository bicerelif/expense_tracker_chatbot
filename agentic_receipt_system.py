"""
Agentic Receipt Management System
A conversational AI agent that can process receipts, analyze expenses, and answer questions.
"""

import os
import sys

# Suppress gRPC warnings BEFORE any imports
os.environ['GRPC_VERBOSITY'] = 'ERROR'
os.environ['GLOG_minloglevel'] = '2'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import sqlite3
import json
from pathlib import Path
from typing import TypedDict, Annotated, List, Dict, Any
import operator
import warnings
import logging

warnings.filterwarnings('ignore')
logging.getLogger('google').setLevel(logging.ERROR)

# LangChain and LangGraph imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

# Import our existing modules
from receipt_parser import ReceiptParser
from visualize_expenses import ExpenseVisualizer


# Define the agent state
class AgentState(TypedDict):
    messages: Annotated[List, operator.add]
    receipt_results: Dict[str, Any]
    visualization_results: List[str]


# Define tools for the agent
@tool
def process_new_receipts(folder_path: str = "receipts") -> str:
    """
    Process all new receipt images in the receipts folder.
    Extracts items, prices, and dates from receipts and saves to database.
    Automatically skips duplicate receipts.
    
    Args:
        folder_path: Path to folder containing receipt images (default: "receipts")
    
    Returns:
        Summary of processing results
    """
    from io import StringIO
    
    try:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            return "❌ Error: GEMINI_API_KEY environment variable not set"
        
        # Capture processing results
        folder = Path(folder_path)
        if not folder.exists():
            return f"❌ Folder '{folder_path}' does not exist. Please create it and add receipt images."
        
        image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}
        image_files = [f for f in folder.iterdir() 
                      if f.is_file() and f.suffix.lower() in image_extensions]
        
        if not image_files:
            return f"❌ No receipt images found in '{folder_path}' folder. Please add receipt images (jpg, png, etc.)"
        
        # Initialize parser with output suppression
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = StringIO()
        sys.stderr = StringIO()
        
        error_messages = []
        
        try:
            # Direct import to avoid re-initialization issues
            import google.generativeai as genai
            import hashlib
            from PIL import Image
            
            # Configure Gemini
            genai.configure(api_key=api_key)
            
            # Get available models
            available_models = []
            try:
                for m in genai.list_models():
                    if 'generateContent' in m.supported_generation_methods:
                        available_models.append(m.name)
            except Exception as e:
                sys.stdout = old_stdout
                sys.stderr = old_stderr
                return f"❌ Error connecting to Gemini API: {str(e)}\nPlease check your API key and internet connection."
            
            # Select best model
            model_preferences = [
                'models/gemini-2.0-flash-exp',
                'models/gemini-2.0-flash',
                'models/gemini-1.5-flash',
                'models/gemini-1.5-flash-latest',
                'models/gemini-pro-vision',
            ]
            
            selected_model = None
            for model_name in model_preferences:
                if model_name in available_models:
                    selected_model = model_name
                    break
            
            if not selected_model and available_models:
                selected_model = available_models[0]
            
            if not selected_model:
                sys.stdout = old_stdout
                sys.stderr = old_stderr
                available_list = '\n  - '.join(available_models[:5])
                return f"❌ No suitable Gemini model found.\n\nAvailable models:\n  - {available_list}\n\nPlease check your API key has access to vision models."
            
            model = genai.GenerativeModel(selected_model)
            
            # Setup database
            db_path = "receipts.db"
            conn = sqlite3.connect(db_path)
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
            
            # Process receipts
            processed = 0
            skipped = 0
            errors = 0
            
            for image_file in image_files:
                try:
                    # Calculate hash
                    sha256_hash = hashlib.sha256()
                    with open(image_file, "rb") as f:
                        for byte_block in iter(lambda: f.read(4096), b""):
                            sha256_hash.update(byte_block)
                    image_hash = sha256_hash.hexdigest()
                    
                    # Check for duplicate
                    conn = sqlite3.connect(db_path)
                    cursor = conn.cursor()
                    cursor.execute("SELECT id FROM receipts WHERE image_hash = ?", (image_hash,))
                    if cursor.fetchone():
                        conn.close()
                        skipped += 1
                        continue
                    conn.close()
                    
                    # Process image
                    try:
                        img = Image.open(image_file)
                    except Exception as e:
                        errors += 1
                        error_messages.append(f"{image_file.name}: Cannot open image - {str(e)[:50]}")
                        continue
                    
                    prompt = """
Analyze this receipt image and extract the following information:
1. The date from the receipt (format: YYYY-MM-DD)
2. All food/product items with their prices

CRITICAL RULES FOR ITEM NAMES:
- Extract ONLY the generic product category name (e.g., "Bread", "Milk", "Eggs", "Tomato")
- Remove ALL brand names, company names, and product line names
- Remove ALL weight/quantity indicators (kg, g, ml, l, pieces, pack, etc.)
- Remove ALL descriptive adjectives (fresh, organic, sliced, etc.)
- Use singular form for countable items (e.g., "Tomato" not "Tomatoes")

ITEMS TO EXCLUDE:
- Discounts, promotions, or savings
- Deposits or refunds
- Bags, packaging
- Non-food items
- Tax lines, subtotals, totals

Return JSON:
{
    "date": "2024-10-04",
    "items": [
        {"item_name": "Bread", "price": 1.80},
        {"item_name": "Milk", "price": 0.71}
    ]
}
"""
                    
                    try:
                        response = model.generate_content([prompt, img])
                        text = response.text.strip()
                    except Exception as e:
                        errors += 1
                        error_messages.append(f"{image_file.name}: API error - {str(e)[:50]}")
                        continue
                    
                    # Clean JSON
                    if text.startswith("```json"):
                        text = text[7:]
                    if text.startswith("```"):
                        text = text[3:]
                    if text.endswith("```"):
                        text = text[:-3]
                    text = text.strip()
                    
                    try:
                        data = json.loads(text)
                    except json.JSONDecodeError as e:
                        errors += 1
                        error_messages.append(f"{image_file.name}: Invalid JSON response - {str(e)[:30]}")
                        continue
                    
                    receipt_date = data.get('date') or 'UNKNOWN'
                    items = data.get('items', [])
                    
                    if items:
                        # Save to database
                        conn = sqlite3.connect(db_path)
                        cursor = conn.cursor()
                        
                        cursor.execute(
                            "INSERT INTO receipts (image_path, image_hash, receipt_date) VALUES (?, ?, ?)",
                            (str(image_file), image_hash, receipt_date)
                        )
                        receipt_id = cursor.lastrowid
                        
                        for item in items:
                            cursor.execute(
                                "INSERT INTO items (receipt_id, item_name, price) VALUES (?, ?, ?)",
                                (receipt_id, item['item_name'], item['price'])
                            )
                        
                        conn.commit()
                        conn.close()
                        processed += 1
                    else:
                        errors += 1
                        error_messages.append(f"{image_file.name}: No items extracted from receipt")
                        
                except Exception as e:
                    errors += 1
                    error_messages.append(f"{image_file.name}: Unexpected error - {str(e)[:50]}")
                    continue
            
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
        
        total = len(image_files)
        summary = f"📊 Processing Summary:\n"
        summary += f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        summary += f"  • Total receipts found: {total}\n"
        summary += f"  • ✅ New receipts processed: {processed}\n"
        summary += f"  • ⊘ Duplicates skipped: {skipped}\n"
        
        if errors > 0:
            summary += f"  • ⚠️ Errors encountered: {errors}\n"
        
        if error_messages:
            summary += f"\n⚠️ Error Details:\n"
            for msg in error_messages[:5]:  # Show first 5 errors
                summary += f"  - {msg}\n"
            if len(error_messages) > 5:
                summary += f"  ... and {len(error_messages) - 5} more errors\n"
        
        if processed > 0:
            summary += f"\n✅ Successfully processed {processed} new receipt(s)!"
        elif skipped == total:
            summary += f"\nℹ️ All receipts have already been processed."
        elif errors > 0:
            summary += f"\n⚠️ Some receipts could not be processed. Check error details above."
        
        return summary
    
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        return f"❌ CRITICAL ERROR processing receipts:\n\n{str(e)}\n\nStack trace:\n{error_detail}"


@tool
def generate_expense_visualizations() -> str:
    """
    Generate comprehensive expense visualization charts from the database.
    Creates 7 different charts analyzing spending patterns.
    
    Returns:
        Summary of generated visualizations
    """
    from io import StringIO
    
    try:
        # Suppress verbose output
        old_stdout = sys.stdout
        sys.stdout = StringIO()
        
        try:
            visualizer = ExpenseVisualizer()
            visualizer.load_data()
            
            charts = []
            
            try:
                visualizer.plot_spending_by_receipt()
                charts.append("spending_by_receipt.png")
            except: pass
            
            try:
                visualizer.plot_top_items_by_frequency()
                charts.append("top_items_frequency.png")
            except: pass
            
            try:
                visualizer.plot_top_items_by_spending()
                charts.append("top_items_spending.png")
            except: pass
            
            try:
                visualizer.plot_spending_over_time()
                charts.append("spending_over_time.png")
            except: pass
            
            try:
                visualizer.plot_spending_by_month()
                charts.append("spending_by_month.png")
            except: pass
            
            try:
                visualizer.plot_price_distribution()
                charts.append("price_distribution.png")
            except: pass
            
            try:
                visualizer.plot_items_per_receipt()
                charts.append("items_per_receipt.png")
            except: pass
        finally:
            sys.stdout = old_stdout
        
        if charts:
            result = f"📊 Generated {len(charts)} visualization chart(s):\n\n"
            for i, chart in enumerate(charts, 1):
                result += f"  {i}. {chart}\n"
            result += f"\n✅ All charts saved successfully!"
            return result
        else:
            return "⚠️ No visualizations could be generated. Please ensure receipts have been processed first."
    
    except FileNotFoundError:
        return "❌ Database not found. Please process some receipts first using the process_new_receipts tool."
    except Exception as e:
        import traceback
        return f"❌ Error generating visualizations:\n\n{str(e)}\n\n{traceback.format_exc()[:400]}"


@tool
def get_expense_summary() -> str:
    """
    Get a comprehensive summary of all expenses from the database.
    Includes total spending, averages, top items, and date ranges.
    
    Returns:
        Detailed expense summary statistics
    """
    try:
        db_path = "receipts.db"
        if not Path(db_path).exists():
            return "No data available. Please process some receipts first."
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get overall statistics
        cursor.execute("SELECT COUNT(*) FROM receipts")
        total_receipts = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*), SUM(price), AVG(price) FROM items")
        total_items, total_spending, avg_price = cursor.fetchone()
        
        # Get average receipt total
        cursor.execute("""
            SELECT AVG(receipt_total) FROM (
                SELECT receipt_id, SUM(price) as receipt_total 
                FROM items GROUP BY receipt_id
            )
        """)
        avg_receipt = cursor.fetchone()[0]
        
        # Get top items by frequency
        cursor.execute("""
            SELECT item_name, COUNT(*) as count 
            FROM items 
            GROUP BY item_name 
            ORDER BY count DESC 
            LIMIT 5
        """)
        top_items = cursor.fetchall()
        
        # Get most expensive items
        cursor.execute("""
            SELECT item_name, price 
            FROM items 
            ORDER BY price DESC 
            LIMIT 5
        """)
        expensive_items = cursor.fetchall()
        
        conn.close()
        
        summary = f"""
📊 EXPENSE SUMMARY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📈 Overall Statistics:
  • Total Receipts: {total_receipts}
  • Total Items: {total_items}
  • Total Spending: ${total_spending:.2f}
  • Average per Receipt: ${avg_receipt:.2f}
  • Average Item Price: ${avg_price:.2f}

🏆 Top 5 Most Purchased Items:
"""
        for idx, (item, count) in enumerate(top_items, 1):
            summary += f"  {idx}. {item}: {count} times\n"
        
        summary += "\n💰 Top 5 Most Expensive Items:\n"
        for idx, (item, price) in enumerate(expensive_items, 1):
            summary += f"  {idx}. {item}: ${price:.2f}\n"
        
        return summary.strip()
    
    except Exception as e:
        import traceback
        return f"❌ Error getting expense summary:\n\n{str(e)}\n\n{traceback.format_exc()[:400]}"


@tool
def query_specific_item(item_name: str) -> str:
    """
    Query information about a specific item from the database.
    Shows how many times it was purchased, total spent, and average price.
    
    Args:
        item_name: Name of the item to query (e.g., "Milk", "Bread")
    
    Returns:
        Detailed information about the item
    """
    try:
        db_path = "receipts.db"
        if not Path(db_path).exists():
            return "No data available. Please process some receipts first."
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Search for items (case-insensitive, partial match)
        cursor.execute("""
            SELECT item_name, COUNT(*), SUM(price), AVG(price), MIN(price), MAX(price)
            FROM items 
            WHERE LOWER(item_name) LIKE LOWER(?)
            GROUP BY item_name
        """, (f"%{item_name}%",))
        
        results = cursor.fetchall()
        conn.close()
        
        if not results:
            return f"No items found matching '{item_name}'. Try a different search term."
        
        response = f"🔍 Items matching '{item_name}':\n\n"
        
        for item, count, total, avg, min_price, max_price in results:
            response += f"📦 {item}\n"
            response += f"  • Purchased: {count} times\n"
            response += f"  • Total spent: ${total:.2f}\n"
            response += f"  • Average price: ${avg:.2f}\n"
            response += f"  • Price range: ${min_price:.2f} - ${max_price:.2f}\n\n"
        
        return response.strip()
    
    except Exception as e:
        import traceback
        return f"❌ Error querying item:\n\n{str(e)}\n\n{traceback.format_exc()[:400]}"


@tool
def get_monthly_spending(year: int = None, month: int = None) -> str:
    """
    Get spending information for a specific month or all months.
    
    Args:
        year: Year to query (e.g., 2024). If None, returns all months.
        month: Month to query (1-12). If None, returns all months in the year.
    
    Returns:
        Monthly spending breakdown
    """
    try:
        db_path = "receipts.db"
        if not Path(db_path).exists():
            return "No data available. Please process some receipts first."
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        if year and month:
            cursor.execute("""
                SELECT strftime('%Y-%m', r.receipt_date) as month, 
                       SUM(i.price) as total, 
                       COUNT(DISTINCT r.id) as receipt_count
                FROM items i
                JOIN receipts r ON i.receipt_id = r.id
                WHERE strftime('%Y', r.receipt_date) = ? 
                  AND strftime('%m', r.receipt_date) = ?
                  AND r.receipt_date != 'UNKNOWN'
                GROUP BY month
            """, (str(year), f"{month:02d}"))
        elif year:
            cursor.execute("""
                SELECT strftime('%Y-%m', r.receipt_date) as month, 
                       SUM(i.price) as total, 
                       COUNT(DISTINCT r.id) as receipt_count
                FROM items i
                JOIN receipts r ON i.receipt_id = r.id
                WHERE strftime('%Y', r.receipt_date) = ?
                  AND r.receipt_date != 'UNKNOWN'
                GROUP BY month
                ORDER BY month
            """, (str(year),))
        else:
            cursor.execute("""
                SELECT strftime('%Y-%m', r.receipt_date) as month, 
                       SUM(i.price) as total, 
                       COUNT(DISTINCT r.id) as receipt_count
                FROM items i
                JOIN receipts r ON i.receipt_id = r.id
                WHERE r.receipt_date != 'UNKNOWN'
                GROUP BY month
                ORDER BY month
            """)
        
        results = cursor.fetchall()
        conn.close()
        
        if not results:
            return "No spending data found for the specified period."
        
        response = "📅 Monthly Spending Breakdown:\n\n"
        total_all = 0
        
        for month, total, receipt_count in results:
            response += f"{month}: ${total:.2f} ({receipt_count} receipts)\n"
            total_all += total
        
        if len(results) > 1:
            response += f"\nTotal: ${total_all:.2f}"
        
        return response
    
    except Exception as e:
        import traceback
        return f"❌ Error getting monthly spending:\n\n{str(e)}\n\n{traceback.format_exc()[:400]}"


# Create the agent
class ReceiptAgent:
    def __init__(self):
        """Initialize the Receipt Management Agent."""
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set")
        
        # Initialize the LLM
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-exp",
            google_api_key=api_key,
            temperature=0.7
        )
        
        # Define available tools
        self.tools = [
            process_new_receipts,
            generate_expense_visualizations,
            get_expense_summary,
            query_specific_item,
            get_monthly_spending
        ]
        
        # Bind tools to LLM
        self.llm_with_tools = self.llm.bind_tools(self.tools)
        
        # Create the graph
        self.graph = self.create_graph()
    
    def create_graph(self):
        """Create the LangGraph workflow."""
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("agent", self.call_model)
        workflow.add_node("tools", ToolNode(self.tools))
        
        # Add edges
        workflow.set_entry_point("agent")
        workflow.add_conditional_edges(
            "agent",
            self.should_continue,
            {
                "continue": "tools",
                "end": END
            }
        )
        workflow.add_edge("tools", "agent")
        
        return workflow.compile()
    
    def call_model(self, state: AgentState):
        """Call the language model with the current state."""
        messages = state["messages"]
        
        # Add system message if this is the first call
        if len(messages) == 1:
            system_msg = SystemMessage(content="""You are a helpful Receipt Management Assistant. You help users:
1. Process receipt images and extract items, prices, and dates
2. Analyze spending patterns and generate insights
3. Answer questions about expenses
4. Create visualizations of spending data

You have access to several tools to help users manage their receipts and expenses.
Be friendly, helpful, and provide clear explanations. When users ask questions about their spending,
use the appropriate tools to get accurate data from the database.

IMPORTANT: When tools return error messages or detailed results, share them with the user in full.
Always suggest relevant actions based on the conversation context.""")
            messages = [system_msg] + messages
        
        response = self.llm_with_tools.invoke(messages)
        return {"messages": [response]}
    
    def should_continue(self, state: AgentState):
        """Determine if the agent should continue or end."""
        last_message = state["messages"][-1]
        
        # If there are no tool calls, we're done
        if not hasattr(last_message, "tool_calls") or not last_message.tool_calls:
            return "end"
        
        return "continue"
    
    def chat(self, user_input: str, debug: bool = False):
        """Process user input and generate response."""
        initial_state = {
            "messages": [HumanMessage(content=user_input)],
            "receipt_results": {},
            "visualization_results": []
        }
        
        try:
            result = self.graph.invoke(initial_state)
            
            # Debug: Show all messages if enabled
            if debug:
                print("\n[DEBUG] All messages:")
                for i, message in enumerate(result["messages"]):
                    print(f"  {i}: {type(message).__name__}")
                    if hasattr(message, "tool_calls") and message.tool_calls:
                        print(f"     Tool calls: {message.tool_calls}")
                    if hasattr(message, "content"):
                        print(f"     Content: {message.content[:100]}...")
            
            # Return the last AI message
            for message in reversed(result["messages"]):
                if isinstance(message, AIMessage):
                    return message.content
            
            return "I'm not sure how to respond to that."
        except Exception as e:
            import traceback
            error_detail = traceback.format_exc()
            return f"❌ I encountered an error:\n\n{str(e)}\n\nTechnical details:\n{error_detail[:600]}\n\nPlease try rephrasing your request."
    
    def run_interactive(self):
        """Run the agent in interactive mode."""
        print("\n" + "="*60)
        print("🤖 RECEIPT MANAGEMENT AGENT")
        print("="*60)
        print("\nI'm your Receipt Management Assistant! I can help you:")
        print("  • Process receipt images")
        print("  • Analyze your spending")
        print("  • Generate visualizations")
        print("  • Answer questions about expenses")
        print("\nType 'quit' or 'exit' to end the conversation.")
        print("="*60 + "\n")
        
        # Quick system check
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            print("⚠️  Warning: GEMINI_API_KEY not found!")
            print("Please set it: export GEMINI_API_KEY='your-api-key'\n")
        
        receipts_folder = Path("receipts")
        if receipts_folder.exists():
            image_count = len([f for f in receipts_folder.iterdir() 
                             if f.suffix.lower() in {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}])
            print(f"📁 Found {image_count} image(s) in receipts folder\n")
        else:
            print("📁 'receipts' folder not found - will be created when needed\n")
        
        while True:
            try:
                user_input = input("You: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'bye', 'goodbye']:
                    print("\n👋 Goodbye! Have a great day!")
                    break
                
                if not user_input:
                    continue
                
                print("\n🤖 Agent: ", end="", flush=True)
                response = self.chat(user_input)
                print(response + "\n")
            
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                import traceback
                print(f"\n❌ Error: {e}")
                print(f"Details: {traceback.format_exc()[:400]}\n")


def main():
    """Main entry point for the agentic system."""
    try:
        agent = ReceiptAgent()
        agent.run_interactive()
    except ValueError as e:
        print(f"❌ Error: {e}")
        print("Please set GEMINI_API_KEY environment variable")
    except Exception as e:
        import traceback
        print(f"❌ Error initializing agent: {e}")
        print(f"Details: {traceback.format_exc()[:600]}")


if __name__ == "__main__":
    main()