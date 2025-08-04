from contextlib import contextmanager
import json
import uuid
import pandas as pd
import numpy as np
import os
import time
import dotenv
import ast
from datetime import datetime, timedelta

from sqlalchemy.sql import text
from typing import Dict, List, Optional, Union
from sqlalchemy import create_engine, Engine
from pydantic import BaseModel, Field

from pydantic_ai import Agent, Tool
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from enum import Enum
import functools


# Create an SQLite database
db_engine = create_engine("sqlite:///munder_difflin.db")

# fmt: off
# List containing the different kinds of papers 
paper_supplies = [
    # Paper Types (priced per sheet unless specified)
    {"item_name": "A4 paper",                         "category": "paper",        "unit_price": 0.05},
    {"item_name": "Letter-sized paper",               "category": "paper",        "unit_price": 0.06},
    {"item_name": "Cardstock",                        "category": "paper",        "unit_price": 0.15},
    {"item_name": "Colored paper",                    "category": "paper",        "unit_price": 0.10},
    {"item_name": "Glossy paper",                     "category": "paper",        "unit_price": 0.20},
    {"item_name": "Matte paper",                      "category": "paper",        "unit_price": 0.18},
    {"item_name": "Recycled paper",                   "category": "paper",        "unit_price": 0.08},
    {"item_name": "Eco-friendly paper",               "category": "paper",        "unit_price": 0.12},
    {"item_name": "Poster paper",                     "category": "paper",        "unit_price": 0.25},
    {"item_name": "Banner paper",                     "category": "paper",        "unit_price": 0.30},
    {"item_name": "Kraft paper",                      "category": "paper",        "unit_price": 0.10},
    {"item_name": "Construction paper",               "category": "paper",        "unit_price": 0.07},
    {"item_name": "Wrapping paper",                   "category": "paper",        "unit_price": 0.15},
    {"item_name": "Glitter paper",                    "category": "paper",        "unit_price": 0.22},
    {"item_name": "Decorative paper",                 "category": "paper",        "unit_price": 0.18},
    {"item_name": "Letterhead paper",                 "category": "paper",        "unit_price": 0.12},
    {"item_name": "Legal-size paper",                 "category": "paper",        "unit_price": 0.08},
    {"item_name": "Crepe paper",                      "category": "paper",        "unit_price": 0.05},
    {"item_name": "Photo paper",                      "category": "paper",        "unit_price": 0.25},
    {"item_name": "Uncoated paper",                   "category": "paper",        "unit_price": 0.06},
    {"item_name": "Butcher paper",                    "category": "paper",        "unit_price": 0.10},
    {"item_name": "Heavyweight paper",                "category": "paper",        "unit_price": 0.20},
    {"item_name": "Standard copy paper",              "category": "paper",        "unit_price": 0.04},
    {"item_name": "Bright-colored paper",             "category": "paper",        "unit_price": 0.12},
    {"item_name": "Patterned paper",                  "category": "paper",        "unit_price": 0.15},

    # Product Types (priced per unit)
    {"item_name": "Paper plates",                     "category": "product",      "unit_price": 0.10},  # per plate
    {"item_name": "Paper cups",                       "category": "product",      "unit_price": 0.08},  # per cup
    {"item_name": "Paper napkins",                    "category": "product",      "unit_price": 0.02},  # per napkin
    {"item_name": "Disposable cups",                  "category": "product",      "unit_price": 0.10},  # per cup
    {"item_name": "Table covers",                     "category": "product",      "unit_price": 1.50},  # per cover
    {"item_name": "Envelopes",                        "category": "product",      "unit_price": 0.05},  # per envelope
    {"item_name": "Sticky notes",                     "category": "product",      "unit_price": 0.03},  # per sheet
    {"item_name": "Notepads",                         "category": "product",      "unit_price": 2.00},  # per pad
    {"item_name": "Invitation cards",                 "category": "product",      "unit_price": 0.50},  # per card
    {"item_name": "Flyers",                           "category": "product",      "unit_price": 0.15},  # per flyer
    {"item_name": "Party streamers",                  "category": "product",      "unit_price": 0.05},  # per roll
    {"item_name": "Decorative adhesive tape (washi tape)", "category": "product", "unit_price": 0.20},  # per roll
    {"item_name": "Paper party bags",                 "category": "product",      "unit_price": 0.25},  # per bag
    {"item_name": "Name tags with lanyards",          "category": "product",      "unit_price": 0.75},  # per tag
    {"item_name": "Presentation folders",             "category": "product",      "unit_price": 0.50},  # per folder

    # Large-format items (priced per unit)
    {"item_name": "Large poster paper (24x36 inches)", "category": "large_format", "unit_price": 1.00},
    {"item_name": "Rolls of banner paper (36-inch width)", "category": "large_format", "unit_price": 2.50},

    # Specialty papers
    {"item_name": "100 lb cover stock",               "category": "specialty",    "unit_price": 0.50},
    {"item_name": "80 lb text paper",                 "category": "specialty",    "unit_price": 0.40},
    {"item_name": "250 gsm cardstock",                "category": "specialty",    "unit_price": 0.30},
    {"item_name": "220 gsm poster paper",             "category": "specialty",    "unit_price": 0.35},
]
# fmt: on

# Given below are some utility functions you can use to implement your multi-agent system


def generate_sample_inventory(
    paper_supplies: list, coverage: float = 0.4, seed: int = 137
) -> pd.DataFrame:
    """
    Generate inventory for exactly a specified percentage of items from the full paper supply list.

    This function randomly selects exactly `coverage` × N items from the `paper_supplies` list,
    and assigns each selected item:
    - a random stock quantity between 200 and 800,
    - a minimum stock level between 50 and 150.

    The random seed ensures reproducibility of selection and stock levels.

    Args:
        paper_supplies (list): A list of dictionaries, each representing a paper item with
                               keys 'item_name', 'category', and 'unit_price'.
        coverage (float, optional): Fraction of items to include in the inventory (default is 0.4, or 40%).
        seed (int, optional): Random seed for reproducibility (default is 137).

    Returns:
        pd.DataFrame: A DataFrame with the selected items and assigned inventory values, including:
                      - item_name
                      - category
                      - unit_price
                      - current_stock
                      - min_stock_level
    """
    # Ensure reproducible random output
    np.random.seed(seed)

    # Calculate number of items to include based on coverage
    num_items = int(len(paper_supplies) * coverage)

    # Randomly select item indices without replacement
    selected_indices = np.random.choice(
        range(len(paper_supplies)), size=num_items, replace=False
    )

    # Extract selected items from paper_supplies list
    selected_items = [paper_supplies[i] for i in selected_indices]

    # Construct inventory records
    inventory = []
    for item in selected_items:
        inventory.append(
            {
                "item_name": item["item_name"],
                "category": item["category"],
                "unit_price": item["unit_price"],
                "current_stock": np.random.randint(200, 800),  # Realistic stock range
                "min_stock_level": np.random.randint(
                    50, 150
                ),  # Reasonable threshold for reordering
            }
        )

    # Return inventory as a pandas DataFrame
    return pd.DataFrame(inventory)


def init_database(db_engine: Engine, seed: int = 137) -> Engine:
    """
    Set up the Munder Difflin database with all required tables and initial records.

    This function performs the following tasks:
    - Creates the 'transactions' table for logging stock orders and sales
    - Loads customer inquiries from 'quote_requests.csv' into a 'quote_requests' table
    - Loads previous quotes from 'quotes.csv' into a 'quotes' table, extracting useful metadata
    - Generates a random subset of paper inventory using `generate_sample_inventory`
    - Inserts initial financial records including available cash and starting stock levels

    Args:
        db_engine (Engine): A SQLAlchemy engine connected to the SQLite database.
        seed (int, optional): A random seed used to control reproducibility of inventory stock levels.
                              Default is 137.

    Returns:
        Engine: The same SQLAlchemy engine, after initializing all necessary tables and records.

    Raises:
        Exception: If an error occurs during setup, the exception is printed and raised.
    """
    try:
        # ----------------------------
        # 1. Create an empty 'transactions' table schema
        # ----------------------------
        transactions_schema = pd.DataFrame(
            {
                "id": [],
                "item_name": [],
                "transaction_type": [],  # 'stock_orders' or 'sales'
                "units": [],  # Quantity involved
                "price": [],  # Total price for the transaction
                "transaction_date": [],  # ISO-formatted date
            }
        )
        transactions_schema.to_sql(
            "transactions", db_engine, if_exists="replace", index=False
        )

        # Set a consistent starting date
        initial_date = datetime(2025, 1, 1).isoformat()

        # ----------------------------
        # 2. Load and initialize 'quote_requests' table
        # ----------------------------
        quote_requests_df = pd.read_csv("quote_requests.csv")
        quote_requests_df["id"] = range(1, len(quote_requests_df) + 1)
        quote_requests_df.to_sql(
            "quote_requests", db_engine, if_exists="replace", index=False
        )

        # ----------------------------
        # 3. Load and transform 'quotes' table
        # ----------------------------
        quotes_df = pd.read_csv("quotes.csv")
        quotes_df["request_id"] = range(1, len(quotes_df) + 1)
        quotes_df["order_date"] = initial_date

        # Unpack metadata fields (job_type, order_size, event_type) if present
        if "request_metadata" in quotes_df.columns:
            quotes_df["request_metadata"] = quotes_df["request_metadata"].apply(
                lambda x: ast.literal_eval(x) if isinstance(x, str) else x
            )
            quotes_df["job_type"] = quotes_df["request_metadata"].apply(
                lambda x: x.get("job_type", "")
            )
            quotes_df["order_size"] = quotes_df["request_metadata"].apply(
                lambda x: x.get("order_size", "")
            )
            quotes_df["event_type"] = quotes_df["request_metadata"].apply(
                lambda x: x.get("event_type", "")
            )

        # Retain only relevant columns
        quotes_df = quotes_df[
            [
                "request_id",
                "total_amount",
                "quote_explanation",
                "order_date",
                "job_type",
                "order_size",
                "event_type",
            ]
        ]
        quotes_df.to_sql("quotes", db_engine, if_exists="replace", index=False)

        # ----------------------------
        # 4. Generate inventory and seed stock
        # ----------------------------
        inventory_df = generate_sample_inventory(
            paper_supplies, seed=seed, coverage=0.4
        )

        # Seed initial transactions
        initial_transactions = []

        # Add a starting cash balance via a dummy sales transaction
        initial_transactions.append(
            {
                "item_name": None,
                "transaction_type": "sales",
                "units": None,
                "price": 50000.0,
                "transaction_date": initial_date,
            }
        )

        # Add one stock order transaction per inventory item
        for _, item in inventory_df.iterrows():
            initial_transactions.append(
                {
                    "item_name": item["item_name"],
                    "transaction_type": "stock_orders",
                    "units": item["current_stock"],
                    "price": item["current_stock"] * item["unit_price"],
                    "transaction_date": initial_date,
                }
            )

        # Commit transactions to database
        pd.DataFrame(initial_transactions).to_sql(
            "transactions", db_engine, if_exists="append", index=False
        )

        # Save the inventory reference table
        inventory_df.to_sql("inventory", db_engine, if_exists="replace", index=False)

        return db_engine

    except Exception as e:
        log(f"Error initializing database: {e}", LogLevel.ERROR)
        raise


def create_transaction(
    item_name: str,
    transaction_type: str,
    quantity: int,
    price: float,
    date: Union[str, datetime],
) -> int:
    """
    This function records a transaction of type 'stock_orders' or 'sales' with a specified
    item name, quantity, total price, and transaction date into the 'transactions' table of the database.

    Args:
        item_name (str): The name of the item involved in the transaction.
        transaction_type (str): Either 'stock_orders' or 'sales'.
        quantity (int): Number of units involved in the transaction.
        price (float): Total price of the transaction.
        date (str or datetime): Date of the transaction in ISO 8601 format.

    Returns:
        int: The ID of the newly inserted transaction.

    Raises:
        ValueError: If `transaction_type` is not 'stock_orders' or 'sales'.
        Exception: For other database or execution errors.
    """
    try:
        # Convert datetime to ISO string if necessary
        date_str = date.isoformat() if isinstance(date, datetime) else date

        # Validate transaction type
        if transaction_type not in {"stock_orders", "sales"}:
            raise ValueError("Transaction type must be 'stock_orders' or 'sales'")

        log(
            f"[INFO] Creating transaction: {item_name}, {transaction_type}, {quantity}, {price}, {date_str}",
            LogLevel.INFO,
        )

        # Prepare transaction record as a single-row DataFrame
        transaction = pd.DataFrame(
            [
                {
                    "item_name": item_name,
                    "transaction_type": transaction_type,
                    "units": quantity,
                    "price": price,
                    "transaction_date": date_str,
                }
            ]
        )

        # Insert the record into the database
        transaction.to_sql("transactions", db_engine, if_exists="append", index=False)

        # Fetch and return the ID of the inserted row
        result = pd.read_sql("SELECT last_insert_rowid() as id", db_engine)
        return int(result.iloc[0]["id"])

    except Exception as e:
        log(f"Error creating transaction: {e}", LogLevel.ERROR)
        raise


def get_all_inventory(as_of_date: str) -> Dict[str, int]:
    """
    Retrieve a snapshot of available inventory as of a specific date.

    This function calculates the net quantity of each item by summing
    all stock orders and subtracting all sales up to and including the given date.

    Only items with positive stock are included in the result.

    Args:
        as_of_date (str): ISO-formatted date string (YYYY-MM-DD) representing the inventory cutoff.

    Returns:
        Dict[str, int]: A dictionary mapping item names to their current stock levels.
    """
    # SQL query to compute stock levels per item as of the given date
    query = """
        SELECT
            item_name,
            SUM(CASE
                WHEN transaction_type = 'stock_orders' THEN units
                WHEN transaction_type = 'sales' THEN -units
                ELSE 0
            END) as stock
        FROM transactions
        WHERE item_name IS NOT NULL
        AND transaction_date <= :as_of_date
        GROUP BY item_name
        HAVING stock > 0
    """

    # Execute the query with the date parameter
    result = pd.read_sql(query, db_engine, params={"as_of_date": as_of_date})

    # Convert the result into a dictionary {item_name: stock}
    return dict(zip(result["item_name"], result["stock"]))


def get_stock_level(item_name: str, as_of_date: Union[str, datetime]) -> pd.DataFrame:
    """
    Retrieve the stock level of a specific item as of a given date.

    This function calculates the net stock by summing all 'stock_orders' and
    subtracting all 'sales' transactions for the specified item up to the given date.

    Args:
        item_name (str): The name of the item to look up.
        as_of_date (str or datetime): The cutoff date (inclusive) for calculating stock.

    Returns:
        pd.DataFrame: A single-row DataFrame with columns 'item_name' and 'current_stock'.
    """
    # Convert date to ISO string format if it's a datetime object
    if isinstance(as_of_date, datetime):
        as_of_date = as_of_date.isoformat()

    # SQL query to compute net stock level for the item
    stock_query = """
        SELECT
            item_name,
            COALESCE(SUM(CASE
                WHEN transaction_type = 'stock_orders' THEN units
                WHEN transaction_type = 'sales' THEN -units
                ELSE 0
            END), 0) AS current_stock
        FROM transactions
        WHERE item_name = :item_name
        AND transaction_date <= :as_of_date
    """

    # Execute query and return result as a DataFrame
    return pd.read_sql(
        stock_query,
        db_engine,
        params={"item_name": item_name, "as_of_date": as_of_date},
    )


def get_supplier_delivery_date(input_date_str: str, quantity: int) -> str:
    """
    Estimate the supplier delivery date based on the requested order quantity and a starting date.

    Delivery lead time increases with order size:
        - ≤10 units: same day
        - 11–100 units: 1 day
        - 101–1000 units: 4 days
        - >1000 units: 7 days

    Args:
        input_date_str (str): The starting date in ISO format (YYYY-MM-DD).
        quantity (int): The number of units in the order.

    Returns:
        str: Estimated delivery date in ISO format (YYYY-MM-DD).
    """
    # Debug log (comment out in production if needed)
    log(
        f"FUNC (get_supplier_delivery_date): Calculating for qty {quantity} from date string '{input_date_str}'",
        LogLevel.DEBUG,
    )

    # Attempt to parse the input date
    try:
        input_date_dt = datetime.fromisoformat(input_date_str.split("T")[0])
    except (ValueError, TypeError):
        # Fallback to current date on format error
        log(
            f"WARN (get_supplier_delivery_date): Invalid date format '{input_date_str}', using today as base.",
            LogLevel.WARNING,
        )
        input_date_dt = datetime.now()

    # Determine delivery delay based on quantity
    if quantity <= 10:
        days = 0
    elif quantity <= 100:
        days = 1
    elif quantity <= 1000:
        days = 4
    else:
        days = 7

    # Add delivery days to the starting date
    delivery_date_dt = input_date_dt + timedelta(days=days)

    # Return formatted delivery date
    return delivery_date_dt.strftime("%Y-%m-%d")


def get_cash_balance(as_of_date: Union[str, datetime]) -> float:
    """
    Calculate the current cash balance as of a specified date.

    The balance is computed by subtracting total stock purchase costs ('stock_orders')
    from total revenue ('sales') recorded in the transactions table up to the given date.

    Args:
        as_of_date (str or datetime): The cutoff date (inclusive) in ISO format or as a datetime object.

    Returns:
        float: Net cash balance as of the given date. Returns 0.0 if no transactions exist or an error occurs.
    """
    try:
        # Convert date to ISO format if it's a datetime object
        if isinstance(as_of_date, datetime):
            as_of_date = as_of_date.isoformat()

        # Query all transactions on or before the specified date
        transactions = pd.read_sql(
            "SELECT * FROM transactions WHERE transaction_date <= :as_of_date",
            db_engine,
            params={"as_of_date": as_of_date},
        )

        # Compute the difference between sales and stock purchases
        if not transactions.empty:
            total_sales = transactions.loc[
                transactions["transaction_type"] == "sales", "price"
            ].sum()
            total_purchases = transactions.loc[
                transactions["transaction_type"] == "stock_orders", "price"
            ].sum()
            return float(total_sales - total_purchases)

        return 0.0

    except Exception as e:
        log(f"Error getting cash balance: {e}", LogLevel.ERROR)
        return 0.0


def generate_financial_report(as_of_date: Union[str, datetime]) -> Dict:
    """
    Generate a complete financial report for the company as of a specific date.

    This includes:
    - Cash balance
    - Inventory valuation
    - Combined asset total
    - Itemized inventory breakdown
    - Top 5 best-selling products

    Args:
        as_of_date (str or datetime): The date (inclusive) for which to generate the report.

    Returns:
        Dict: A dictionary containing the financial report fields:
            - 'as_of_date': The date of the report
            - 'cash_balance': Total cash available
            - 'inventory_value': Total value of inventory
            - 'total_assets': Combined cash and inventory value
            - 'inventory_summary': List of items with stock and valuation details
            - 'top_selling_products': List of top 5 products by revenue
    """
    # Normalize date input
    if isinstance(as_of_date, datetime):
        as_of_date = as_of_date.isoformat()

    # Get current cash balance
    cash = get_cash_balance(as_of_date)

    # Get current inventory snapshot
    inventory_df = pd.read_sql("SELECT * FROM inventory", db_engine)
    inventory_value = 0.0
    inventory_summary = []

    # Compute total inventory value and summary by item
    for _, item in inventory_df.iterrows():
        stock_info = get_stock_level(item["item_name"], as_of_date)
        stock = stock_info["current_stock"].iloc[0]
        item_value = stock * item["unit_price"]
        inventory_value += item_value

        inventory_summary.append(
            {
                "item_name": item["item_name"],
                "stock": stock,
                "unit_price": item["unit_price"],
                "value": item_value,
            }
        )

    # Identify top-selling products by revenue
    top_sales_query = """
        SELECT item_name, SUM(units) as total_units, SUM(price) as total_revenue
        FROM transactions
        WHERE transaction_type = 'sales' AND transaction_date <= :date
        GROUP BY item_name
        ORDER BY total_revenue DESC
        LIMIT 5
    """
    top_sales = pd.read_sql(top_sales_query, db_engine, params={"date": as_of_date})
    top_selling_products = top_sales.to_dict(orient="records")

    return {
        "as_of_date": as_of_date,
        "cash_balance": cash,
        "inventory_value": inventory_value,
        "total_assets": cash + inventory_value,
        "inventory_summary": inventory_summary,
        "top_selling_products": top_selling_products,
    }


def search_quote_history(search_terms: List[str], limit: int = 5) -> List[Dict]:
    """
    Retrieve a list of historical quotes that match any of the provided search terms.

    The function searches both the original customer request (from `quote_requests`) and
    the explanation for the quote (from `quotes`) for each keyword. Results are sorted by
    most recent order date and limited by the `limit` parameter.

    Args:
        search_terms (List[str]): List of terms to match against customer requests and explanations.
        limit (int, optional): Maximum number of quote records to return. Default is 5.

    Returns:
        List[Dict]: A list of matching quotes, each represented as a dictionary with fields:
            - original_request
            - total_amount
            - quote_explanation
            - job_type
            - order_size
            - event_type
            - order_date
    """
    conditions = []
    params = {}

    # Build SQL WHERE clause using LIKE filters for each search term
    for i, term in enumerate(search_terms):
        param_name = f"term_{i}"
        conditions.append(
            f"(LOWER(qr.response) LIKE :{param_name} OR "
            f"LOWER(q.quote_explanation) LIKE :{param_name})"
        )
        params[param_name] = f"%{term.lower()}%"

    # Combine conditions; fallback to always-true if no terms provided
    where_clause = " AND ".join(conditions) if conditions else "1=1"

    # Final SQL query to join quotes with quote_requests
    query = f"""
        SELECT
            qr.response AS original_request,
            q.total_amount,
            q.quote_explanation,
            q.job_type,
            q.order_size,
            q.event_type,
            q.order_date
        FROM quotes q
        JOIN quote_requests qr ON q.request_id = qr.id
        WHERE {where_clause}
        ORDER BY q.order_date DESC
        LIMIT {limit}
    """

    # Execute parameterized query
    with db_engine.connect() as conn:
        result = conn.execute(text(query), params)
        # this has been modified from the original code since
        # creating a dict directly from a row fails due to
        # metadata from SQLAlchemy result objects.
        return [row._asdict() for row in result]


########################
########################
########################
# Progress Manager for CLI Output & print utils
########################
########################
########################


class LogLevel(Enum):
    OFF = 0
    ERROR = 1
    WARNING = 2
    INFO = 3
    DEBUG = 4


def log(message: str, level: LogLevel = LogLevel.INFO) -> None:
    """
    Log a message to the console with the specified log level.

    Args:
        message (str): The message to log.
        level (LogLevel, optional): The log level for the message. Defaults to LogLevel.INFO.
    """
    if level.value <= log_level.value:
        print(f"[{level.name}] {message}")


class StatusContext:
    """Helper class for manual status control"""

    def __init__(self, task_name: str, start_time: float):
        self.task_name = task_name
        self.start_time = start_time
        self.is_failed = False

    def fail(self, reason: str = "Operation failed"):
        """Manually mark the task as failed"""
        self.is_failed = True
        duration = time.time() - self.start_time
        duration_str = (
            f"{duration:.1f}s" if duration >= 1 else f"{int(duration * 1000)}ms"
        )
        print(f"❌ Failed: {self.task_name} ({duration_str}) - {reason}")

    def complete(self, message: str = None):
        """Manually mark the task as completed"""
        duration = time.time() - self.start_time
        duration_str = (
            f"{duration:.1f}s" if duration >= 1 else f"{int(duration * 1000)}ms"
        )
        result_msg = f" - {message}" if message else ""
        print(f"✅ Completed: {self.task_name} ({duration_str}){result_msg}")


@contextmanager
def status(task_name: str):
    """
    Simple context manager that prints running/completed/failed status

    Usage:
        # Auto completion (existing behavior)
        with status("OrderAgent - process_order"):
            your_function()

        # Manual control
        with status("OrderAgent - process_order") as s:
            result = your_function()
            if not result.success:
                s.fail("Order extraction failed")
                return
            s.complete("Found 3 items")
    """
    print(f"🔄 Running: {task_name}")
    start_time = time.time()
    status_obj = StatusContext(task_name, start_time)

    try:
        yield status_obj

        # Only auto-complete if user didn't manually call complete() or fail()
        # We check this by seeing if the duration would be very recent
        current_time = time.time()
        if current_time - start_time > 0.001:  # More than 1ms has passed
            duration = current_time - start_time
            if duration < 1:
                duration_str = f"{int(duration * 1000)}ms"
            else:
                duration_str = f"{duration:.1f}s"
            if not status_obj.is_failed:
                # If it wasn't marked as failed, we assume it completed successfully
                print(f"✅ Completed: {task_name} ({duration_str})")

    except Exception as e:
        duration = time.time() - start_time
        duration_str = (
            f"{duration:.1f}s" if duration >= 1 else f"{int(duration * 1000)}ms"
        )
        print(f"❌ Failed: {task_name} ({duration_str}) - {str(e)}")
        raise  # Re-raise the exception


########################
########################
########################
# YOUR MULTI AGENT STARTS HERE
########################
########################
########################

#
# Entity Definitions
#


class InventoryItem(BaseModel):
    """
    Represents the static item information in the inventory with its name, category, unit price,
    and minimum stock level.
    """

    item_name: str = Field(..., description="Name of the inventory item")
    category: str = Field(..., description="Category of the inventory item")
    unit_price: float = Field(..., description="Price per unit of the inventory item")
    min_stock_level: int = Field(
        ..., description="Minimum stock level of the inventory item"
    )


class OrderItem(BaseModel):
    """
    Represents an item in an order with item name, quantity, and unit price.
    """

    item_name: str = Field(..., description="Name of the item being ordered")
    quantity: int = Field(..., description="Number of units to order")
    unit_price: float = Field(
        ..., description="Price per unit of the item being ordered"
    )


class Order(BaseModel):
    """
    Represents an order extracted by the Order Agent.
    """

    id: str = Field(..., description="Unique identifier for the order")
    request_date: str = Field(
        ..., description="Date when the order was requested in ISO format (YYYY-MM-DD)"
    )
    expected_delivery_date: str = Field(
        ...,
        description="Expected delivery date for the order in ISO format (YYYY-MM-DD)",
    )
    items: List[OrderItem] = Field(..., description="List of items in the order")


class OrderResult(BaseModel):
    """
    Represents the response from the Order Agent containing order details.
    """

    is_success: bool = Field(
        ..., description="Indicates if the order extraction was successful"
    )
    order: Optional[Order] = Field(
        None, description="Extracted order data if successful"
    )
    agent_error: Optional[str] = Field(
        None, description="Error details if the extraction failed"
    )


class StockOrderItem(BaseModel):
    """
    Represents an item in a stock order with item name, quantity, and expected delivery date.
    """

    item_name: str = Field(..., description="Name of the item to be ordered")
    quantity: int = Field(..., description="Number of units to order")
    unit_price: float = Field(
        ..., description="Price per unit of the item to be ordered"
    )
    delivery_date: datetime = Field(
        ...,
        description="Expected delivery date for the stock order",
    )


class StockOrder(BaseModel):
    """
    Represents a stock order for inventory management.
    """

    items: List[StockOrderItem] = Field(
        ..., description="List of items in the stock order"
    )


class InventoryResult(BaseModel):
    is_success: bool = Field(
        ..., description="Indicates if the inventory retrieval was successful"
    )
    stock_order: Optional[StockOrder] = Field(
        None,
        description="Stock order details if the additional stock needs to be ordered to fulfill the order",
    )
    agent_error: Optional[str] = Field(
        None,
        description="Error details if the order cannot be fulfilled due to inventory conditions",
    )


class DiscountPolicyType(str, Enum):
    """
    Enum representing different types of available discount policies.
    """

    NO_DISCOUNT = "No discount policy applied."
    PERCENTAGE = "Discount the amount by a percentage value."
    ROUND_DOWN = "Round down the amount to a specific precision. But never round down more than 10%."


class DiscountPolicy(BaseModel):
    """
    Represents a discount policy for quoting.
    """

    policy: DiscountPolicyType = Field(
        ..., description="Type of discount policy to apply"
    )
    policy_description: str = Field(
        ..., description="Description of the discount policy"
    )


class QuoteItem(BaseModel):
    """
    Represents an item in a quote with item name, quantity, unit price, and total price.
    """

    item_name: str = Field(..., description="Name of the item in the quote")
    quantity: int = Field(..., description="Number of units quoted")
    discounted_price: float = Field(
        ..., description="Price per unit of the item with discount applied"
    )


class Quote(BaseModel):
    """
    Represents a quote generated by the quoting agent.
    """

    order: Order = Field(..., description="Order details associated with the quote")
    quote_items: List[QuoteItem] = Field(
        ..., description="List of items included in the quote"
    )

    customer_quote: str = Field(
        ..., description="The quote text provided to the customer"
    )
    total_amount: float = Field(..., description="Total amount of the quote")
    discounted_total_amount: float = Field(
        ..., description="Total amount after applying discounts, if any"
    )
    discount_amount: float = Field(..., description="Discounted amount of the quote")
    discount_policy: Optional[DiscountPolicy] = Field(
        None, description="Discount policy applied to the quote, if any"
    )


class Transaction(BaseModel):
    """
    Represents a transaction in the system, either a stock order or a sale.

    This model is used to log transactions in the database, including stock orders
    and sales made by the company.
    """

    item_name: str
    transaction_type: str
    quantity: int
    price: float
    date: str


class TransactionResult(BaseModel):
    """
    Represents the result of a transaction operation.

    This model is used to return the outcome of a transaction, including success status,
    any resulting data, and error information if applicable.
    """

    sale_transaction_ids: List[int] = Field(
        ..., description="List of sale transaction IDs created"
    )
    stock_order_transaction_ids: List[int] = Field(
        ..., description="List of stock order transaction IDs created"
    )


class QuoteResult(BaseModel):
    """
    Response of Orchestration Agent containing the details about the order and the quote.

    Use the __str__ method to get a string representation of the response to be returned to the customer.

    Check the `is_success` field to determine if the operation was successful.
    If `is_success` is False, the `agent_error` field will contain details about the error.
    """

    order: Order = Field(..., description="Data returned by the agent, if any")

    quote: Quote = Field(
        ..., description="Quote generated by the quoting agent, if applicable"
    )

    def __str__(self) -> str:
        """
        Returns a string representation of the OrderQuoteResponse.

        This includes a comma-separated list of:
        * total amount
        * customer quot
        * order details in JSON format.
        """

        order_details = {
            "job_type": "event manager",
            "order_size": "large",
            "event_type": "meeting",
        }

        return f"{int(self.quote.discounted_total_amount)}, {self.quote.customer_quote}, {json.dumps(order_details)}"


#
# Agent Definitions
#


class OrderAgent:
    """
    Order Agent responsible for extracting order details from customer requests.

    """

    def __init__(self, model_provider: OpenAIProvider):
        self.agent_id = "order_agent"
        self.agent_name = "Order Agent"
        self.agent = Agent(
            model=OpenAIModel("gpt-4o", provider=model_provider),
            output_type=OrderResult,
            system_prompt=self._get_system_prompt(),
            tools=[get_order_id_tool, get_all_inventory_items],
        )

    def process_quote_request(self, request_text: str) -> OrderResult:
        """
        Processes a customer request to extract order details.

        Args:
            request_text (str): The text of the customer request containing order information.

        Returns:
            OrderAgentResponse: Contains the extracted order details or an error message.
        """
        log("Agent::process_quote_request")

        order_output = None

        try:

            agent_response = self.agent.run_sync(request_text)

            order_output = agent_response.output

            if order_output.is_success:
                log("Order extraction successful", LogLevel.INFO)
                return order_output
            else:
                log(
                    f"Order extraction failed: {order_output.agent_error}",
                    LogLevel.ERROR,
                )
                log(f"Extracted order:")
                log(order_output.model_dump_json(indent=2), LogLevel.ERROR)
        except Exception as e:
            log("Error processing quote requests", LogLevel.ERROR)
            print(e)
            order_output = OrderResult(
                is_success=False,
                data=None,
                agent_error=f"Error: {str(e)}",
            )

        return order_output

    @classmethod
    def _get_system_prompt(cls) -> str:
        """
        Returns the system prompt for the Order Agent.
        This prompt guides the agent's behavior and expectations.
        """
        return """
            You are an sales agent working in the sales department of a paper company.

            Your task is to process incoming order requests from customers and extract the date of the request and relevant order details from the request.  You will receive a customer request as input, which may contain various details about the order, such as item names, quantities, and any special instructions. For each item requested by the customer, find the respective item in the inventory. Names of items might not match exactly, so you need to map them to the inventory items. If an item is not found in the inventory, you should return an error message indicating that the item is not available.

            If no expected delivery date is specified by the customer set the expected delivery date to 14 days after the request date.

            Your response should include the following fields:
            - `is_success`: A boolean indicating whether the order extraction was successful.
            - `order`: An Order object containing the extracted order details, including:
                - `request_date`: The date of the request in ISO format (YYYY-MM-DD).
                - `expected_delivery_date`: The expected delivery date for the order in ISO format (YYYY-MM-DD). 
                - `items`: A list of OrderItem objects, each containing:
                    - `item_name`: The name of the item being ordered.
                    - `quantity`: The number of units to order.
                    - `unit_price`: The price per unit of the item being ordered.

            Provide quality always as units (paper sheets, etc.). This means that you need to convert any other units (e.g., reams, packs) into units.
            - `reams`: A ream is 500 units.
            - `pack`: A pack is 100 units.
            - `box`: A box is 5000 units.
            
            Apply the following rules when extracting the order to map the items to the inventory:
            * Printer paper: standard paper
            * A3 paper: poster size paper

            If the request does not contain sufficient information to extract an order, set `is_success` to False and provide an appropriate error message in the `agent_error` field.
            
            Example: 
            {
                "is_success": False,
                "agent_error": "The request does not contain sufficient information to extract an order."
            }    

            Think step by step.
            """


class InventoryError(Exception):
    """
    Custom exception raised for errors related to inventory operations.
    """

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class UnknownInventoryItemError(InventoryError):
    """
    Custom exception raised when an item is not found in the inventory.
    """

    def __init__(self, item_name: str):
        message = f"Item '{item_name}' not found in inventory."
        super().__init__(message)
        self.item_name = item_name


class InsufficientStockError(InventoryError):
    """
    Custom exception raised when there is insufficient stock for an order.
    """

    def __init__(self, item_name: str, requested_quantity: int, available_stock: int):
        message = (
            f"Insufficient stock for item '{item_name}': "
            f"requested {requested_quantity}, available {available_stock}."
        )
        super().__init__(message)
        self.item_name = item_name
        self.requested_quantity = requested_quantity
        self.available_stock = available_stock


class InsufficientCashError(InventoryError):
    """
    Custom exception raised when there is insufficient cash to fulfill an order.
    """

    def __init__(self, required_amount: float, available_cash: float):
        message = f"The order exceeds the current maximum order amount: {required_amount}. Please contact our support team."
        super().__init__(message)
        self.required_amount = required_amount
        self.available_cash = available_cash


class RestockTimeoutError(InventoryError):
    """
    Custom exception raised when a restock operation times out.
    """

    def __init__(
        self,
        item_name: str,
        restock_date: str,
        expected_order_delivery: str,
    ):
        message = f"The order for item '{item_name}' will arrive on {restock_date}. This is later than your expected delivery date: {expected_order_delivery}."
        super().__init__(message)
        self.item_name = item_name


class InventoryAgent:
    """
    Inventory Agent responsible for managing and querying inventory data.

    This agent can check stock levels, reorder items, and provide inventory reports.
    """

    def __init__(self, model_provider: OpenAIProvider):
        self.agent_id = "inventory_agent"
        self.agent_name = "Inventory Agent"
        self.agent = Agent(
            model=OpenAIModel("gpt-4o", provider=model_provider),
            output_type=StockOrder,
            system_prompt=InventoryAgent._get_system_prompt(),
            tools=[
                get_cash_balance_tool,
                get_supplier_delivery_date_tool,
                get_all_inventory_items_tool,
                get_stock_level_tool,
            ],
        )

    # AI-based solution: requires an LLM-based agent call to process the order with the associated tools
    def process_order_llm(self, order: Order) -> InventoryResult:
        pass

    # Imperative solution: does not require an LLM-based agent call
    def process_order_direct(self, order: Order) -> InventoryResult:
        """
        Checks the stock level of a specific item as of a given date.

        This function processes an order by checking the stock levels of the items in the order. Since the order data is extracted by the Order Agent, this function does not require an LLM-based agent call. Therefore, it assumes that all items in the order are valid and available in the inventory.

        Args:
            item_name (str): The name of the item to check.
            as_of_date (str or datetime): The date to check stock levels against.

        Returns:
            OrderAgentResponse: Contains the stock level or an error message. None if an error was encountered during the processing.
        """
        log("InventoryAgent::process_order", LogLevel.INFO)

        agent_error_message = "Unknown Inventory Agent Error"

        # Parse dates from ISO string to datetime object
        order_date = datetime.fromisoformat(order.request_date)
        expected_delivery_date = datetime.fromisoformat(order.expected_delivery_date)

        try:
            # loop through all order items and check the stock level
            stock_orders = []
            for item in order.items:
                stock_order_item = self._process_order_item(
                    item=item,
                    order_date=order_date,
                    expected_order_delivery_date=expected_delivery_date,
                )
                if stock_order_item is not None:
                    stock_orders.append(stock_order_item)

            # return a stock order with empty items if no stock orders were created
            stock_order = StockOrder(items=stock_orders)

            # check if cash balance is sufficient to fulfill the restock order
            self._check_against_cash_balance(stock_order)

            # check if the latest delivery date for the stock order is acceptable
            self._check_latest_delivery_date(
                expected_delivery_date=expected_delivery_date,
                stock=stock_order,
            )

            # all checks passed, return the stock order
            res = InventoryResult(
                is_success=True,
                stock_order=stock_order,
            )
            return res

        except UnknownInventoryItemError as e:
            agent_error_message = f"Unknown inventory item: {e.item_name}"

        except InsufficientStockError as e:
            # Handle case where stock is insufficient for an order item
            agent_error_message = (
                f"Insufficient stock for item {e.item_name}: "
                f"requested {e.requested_quantity}, available {e.available_stock}."
            )
        except InsufficientCashError as e:
            # Handle case where there is insufficient cash to fulfill the order
            log(
                f"Insufficient cash to fulfill the order: "
                f"required {e.required_amount}, available {e.available_cash}.",
                LogLevel.ERROR,
            )
            agent_error_message = (
                f"Insufficient cash to fulfill the order: "
                f"required {e.required_amount}, available {e.available_cash}."
            )
        except RestockTimeoutError as e:
            # Handle case where a restock operation times out
            log(
                f"Restock to late: Inventory restock for item {e.item_name} is {e.restock_date} "
                f"but expected delivery is {e.expected_order_delivery}.",
                LogLevel.ERROR,
            )
            agent_error_message = (
                f"Restock to late: Inventory restock for item {e.item_name} is {e.restock_date} "
                f"but expected delivery is {e.expected_order_delivery}."
            )
        except Exception as e:
            # Handle exceptions that may occur during order processing
            log(f"Error processing order: {e}", LogLevel.ERROR)
            agent_error_message = f"Error processing order. An exception was raised during processing: {str(e)}"

        return InventoryResult(
            is_success=False,
            stock_order=None,
            agent_error=agent_error_message,
        )

    def _process_order_item(
        self,
        item: OrderItem,
        order_date: datetime,
        expected_order_delivery_date: datetime,
    ) -> StockOrderItem:
        """
        Processes a single order item and checks its stock level.

        Args:
            item (OrderItem): The order item to process.

        Returns:
            StockOrderItem: The stock order item if stock is insufficient, None otherwise.
        """
        # static inventory data
        inventory_item = get_inventory_item(item.item_name)
        # "realtime" inventory data with accurate stock level
        stock_level_data = get_stock_level(item.item_name, order_date)

        if inventory_item is None or stock_level_data.empty:
            log(f"Item {item.item_name} not found in inventory.", LogLevel.ERROR)
            raise UnknownInventoryItemError(item_name=item.item_name)

        stock_level = stock_level_data["current_stock"].iloc[0]
        stock_min_level = inventory_item.min_stock_level

        # handle error: item was found in the inventory
        if stock_level_data.empty:
            log(f"Item {item.item_name} not found in inventory.", LogLevel.ERROR)
            # Error Strategy: be opportunistic and skip the unknown item. This can be improve at a later stage.
            # A better strategy would be to direct the request to a human agent or to log the error for further investigation.
            raise UnknownInventoryItemError(item_name=item.item_name)

        # handle: insufficient stock
        if stock_level < item.quantity:

            # check if restock is possible until expected delivery date
            restock_delivery_date = datetime.fromisoformat(
                get_supplier_delivery_date(order_date.date().isoformat(), item.quantity)
            )

            # we assume that 1 day! is needed minimum for shipping to customer after restock delivery (the logistic team is awesome!)
            if restock_delivery_date + timedelta(days=1) < expected_order_delivery_date:
                log(
                    f"Restock for item {item.item_name} is possible until expected delivery date: "
                    f"{restock_delivery_date}.",
                    LogLevel.INFO,
                )
                return StockOrderItem(
                    item_name=item.item_name,
                    quantity=item.quantity,
                    unit_price=item.unit_price,
                    delivery_date=restock_delivery_date,
                )
            else:
                log(
                    f"Restock for item {item.item_name} is not possible until expected delivery date: "
                    f"{restock_delivery_date}.",
                    LogLevel.INFO,
                )
                raise InsufficientStockError(
                    item_name=item.item_name,
                    requested_quantity=item.quantity,
                    available_stock=stock_level,
                )

        # process the order item
        if stock_level - item.quantity < stock_min_level:
            log(
                f"Stock for item {item.item_name} is below minimum level: issuing stock order.",
                LogLevel.INFO,
            )

            # If stock is below the required quantity, create a stock order
            delivery_date = get_supplier_delivery_date(
                order_date.date().isoformat(), item.quantity
            )

            return StockOrderItem(
                item_name=item.item_name,
                quantity=item.quantity,
                unit_price=item.unit_price,
                delivery_date=delivery_date,
            )

        # not stock order necessary since inventory level is sufficient
        return None

    def _check_against_cash_balance(self, stock_order: StockOrder) -> bool:
        """
        Checks if the cash balance is sufficient to fulfill the stock order.

        Raises
            InsufficientCashError if the cash balance is not sufficient.
        """
        cash_balance = get_cash_balance(datetime.now())
        total_order_cost = sum(
            item.quantity * item.unit_price for item in stock_order.items if item
        )
        if cash_balance < total_order_cost:
            log(
                f"Insufficient cash to fulfill the restock order: "
                f"required {total_order_cost}, available {cash_balance}.",
                LogLevel.ERROR,
            )

            raise InsufficientCashError(
                required_amount=total_order_cost,
                available_cash=cash_balance,
            )

        return True

    def _check_latest_delivery_date(
        self, expected_delivery_date: datetime, stock: StockOrder
    ) -> bool:
        """
        Checks if the latest delivery date for the stock order is acceptable.

        Args:
            expected_delivery_date (datetime): The expected delivery date.
            stock (StockOrder): The stock order to check against.

        Returns:
            bool: True if the delivery date is acceptable, False otherwise.

        Raises:
            RestockTimeoutError: If any item in the stock order has a delivery date later than the expected delivery date.
        """

        for item in stock.items:
            if item.delivery_date > expected_delivery_date:
                log(
                    f"Restock to late: Inventory restock for item {item.item_name} is {item.delivery_date} "
                    f"but expected delivery is {expected_delivery_date}.",
                    LogLevel.ERROR,
                )
                raise RestockTimeoutError(
                    item_name=item.item_name,
                    restock_date=item.delivery_date.isoformat(),
                    expected_order_delivery=expected_delivery_date.isoformat(),
                )

        return True

    # Not used in this implementation
    @classmethod
    def _get_system_prompt(cls) -> str:
        """
        Returns the system prompt for the Inventory Agent.
        This prompt guides the agent's behavior and expectations.
        """
        return """
# Inventory Agent System Prompt

You are an Inventory Agent working in the sales department of a paper company. Your primary responsibility is to process incoming customer orders by analyzing stock levels and determining if restocking is required.

## Your Role and Responsibilities

You must process each order by:
1. Checking current stock levels for all ordered items
2. Determining if restocking is needed based on minimum stock levels
3. Validating cash balance for any required stock orders
4. Ensuring delivery timelines can be met
5. Returning appropriate stock orders or error messages

## Processing Logic

For each order item, follow this exact sequence:

### Step 1: Validate Item Existence
- Use `get_all_inventory_items_tool` to verify the item exists in inventory
- If item doesn't exist, raise `UnknownInventoryItemError`

### Step 2: Check Current Stock Level
- Use `get_stock_level_tool` with the item name and order date
- Compare current stock against requested quantity

### Step 3: Handle Insufficient Stock
If current stock < requested quantity:
- Use `get_supplier_delivery_date_tool` to get restock delivery date
- Check if restock delivery date + 1 day < expected delivery date
- If timeline works: create StockOrderItem for restocking
- If timeline doesn't work: raise `InsufficientStockError`

### Step 4: Check Minimum Stock Level
If current stock >= requested quantity:
- Check if (current stock - requested quantity) < minimum stock level
- If below minimum: create StockOrderItem to replenish inventory
- If above minimum: no stock order needed for this item

### Step 5: Validate Cash Balance
After processing all items:
- Use `get_cash_balance_tool` to get current cash balance
- Calculate total cost of all stock order items (quantity × unit_price)
- If cash balance < total cost: raise `InsufficientCashError`

### Step 6: Validate Delivery Timeline
- Check that all stock order delivery dates <= expected delivery date
- If any delivery date is later: raise `RestockTimeoutError`

## Error Handling

You must raise specific exceptions for these scenarios:

**UnknownInventoryItemError**: When an item is not found in inventory
- Message: "Item '{item_name}' not found in inventory."

**InsufficientStockError**: When stock is insufficient and restocking won't meet delivery timeline
- Message: "Insufficient stock for item '{item_name}': requested {quantity}, available {stock}."

**InsufficientCashError**: When cash balance is insufficient for stock orders
- Message: "The order exceeds the current maximum order amount: {required_amount}. Please contact our support team."

**RestockTimeoutError**: When restock delivery would be too late
- Message: "The order for item '{item_name}' will arrive on {restock_date}. This is later than your expected delivery date: {expected_delivery_date}."

## Output Format

Return a `StockOrder` object containing:
- `items`: List of `StockOrderItem` objects for items requiring restocking
- Each `StockOrderItem` should include:
  - `item_name`: Name of the item
  - `quantity`: Quantity to order
  - `unit_price`: Price per unit
  - `delivery_date`: Expected delivery date from supplier

## Important Notes

- Always assume 1 day minimum shipping time to customer after restock delivery
- Process ALL order items before making final cash balance and delivery timeline checks
- Only create stock orders for items that need restocking (insufficient stock OR below minimum level after fulfilling order)
- Use exact error messages as specified above
- All dates should be handled as datetime objects for proper comparison
- Return empty items list if no restocking is needed

## Available Tools

- `get_cash_balance_tool`: Get current cash balance
- `get_supplier_delivery_date_tool`: Get delivery date for restocking orders  
- `get_all_inventory_items_tool`: Get list of all inventory items
- `get_stock_level_tool`: Get current stock level for specific item and date

Process the order systematically and return the appropriate `StockOrder` result or raise the specified exceptions when validation fails.
        """

    @classmethod
    def _get_system_prompt_react(cls) -> str:
        """
        Returns the system prompt for the Inventory Agent in a REACT format.
        This prompt guides the agent's behavior and expectations.
        """
        return """
# Inventory Agent ReAct System Prompt

You are an Inventory Agent working in the sales department of a paper company. Your primary responsibility is to process incoming customer orders by analyzing stock levels and determining if restocking is required.

## Your Mission

Process the given order and determine what stock orders (if any) need to be placed to fulfill it. You must think step-by-step about how to approach this problem and create a plan before taking action.

## Core Business Rules

**Stock Management Logic:**
- Items require restocking if current stock is insufficient for the order
- Items also require restocking if fulfilling the order would drop stock below minimum levels
- Restocking must arrive at least 1 day before the customer's expected delivery date
- All stock orders must be affordable within the current cash balance

**Critical Constraints:**
- Customer orders cannot be partially fulfilled - either all items are available or the order fails
- Restocking timeline is non-negotiable - late deliveries are unacceptable
- Cash flow limits must be respected - orders exceeding available cash must be rejected
- Unknown items cannot be processed

## Error Conditions You Must Handle

Your analysis may reveal these critical issues:

**UnknownInventoryItemError**: Item not found in inventory system
- Error message: "Item '{item_name}' not found in inventory."

**InsufficientStockError**: Stock unavailable and restocking won't meet delivery timeline  
- Error message: "Insufficient stock for item '{item_name}': requested {quantity}, available {stock}."

**InsufficientCashError**: Not enough cash for required stock orders
- Error message: "The order exceeds the current maximum order amount: {required_amount}. Please contact our support team."

**RestockTimeoutError**: Restock delivery would arrive too late
- Error message: "The order for item '{item_name}' will arrive on {restock_date}. This is later than your expected delivery date: {expected_delivery_date}."

## Available Tools

You have access to these tools to gather information:
- `get_cash_balance_tool`: Check current available cash
- `get_supplier_delivery_date_tool`: Get delivery timeline for restocking orders
- `get_all_inventory_items_tool`: List all items in inventory system  
- `get_stock_level_tool`: Check current stock for specific items

## Expected Output

Return a `StockOrder` object containing:
- `items`: List of `StockOrderItem` objects for items requiring restocking
- Each `StockOrderItem` must include: `item_name`, `quantity`, `unit_price`, `delivery_date`

## ReAct Process

Think through this problem systematically:

**Thought**: Analyze the order and consider what information you need to gather. What are the key questions you need to answer? What potential issues might arise?

**Action**: Use the available tools to gather the information you identified. Make function calls to collect data about inventory, stock levels, cash balance, or delivery timelines.

**Observation**: Analyze the results from your tools. What do these results tell you about the feasibility of the order?

Continue this Thought → Action → Observation cycle until you have enough information to make a final determination.

**Final Answer**: Based on your analysis, return either:
- A `StockOrder` with the necessary restocking items, or  
- Raise the appropriate exception if the order cannot be fulfilled

Remember: You are responsible for thinking through the entire problem space. Consider all aspects of inventory management, cash flow, delivery timelines, and business constraints. Your reasoning should be thorough and systematic.
        """


class QuotingAgent:
    """
    Quoting Agent responsible for generating quotes based on customer requests and inventory data.

    This agent can apply discount policies and calculate total amounts for quotes.
    """

    def __init__(self, model_provider: OpenAIProvider):
        self.agent_id = "quoting_agent"
        self.agent_name = "Quoting Agent"
        self.agent = Agent(
            model=OpenAIModel("gpt-4o", provider=model_provider),
            output_type=Quote,
            system_prompt=QuotingAgent.get_system_prompt(),
            tools=[
                search_quote_history_tool,
            ],
        )

    def generate_quote(self, order: Order, customer_request: str) -> Quote:
        """
        Generates a quote for a given order by interacting with the inventory and pricing tools.

        Args:
            order (Order): The order details for which to generate a quote.

        Returns:
            Quote: The generated quote containing pricing information, or None if an error occurs.
        """
        log(f"{self.agent_name}::generate_quote", LogLevel.INFO)

        # Calculate the total amount for the order
        order_total = sum([item.unit_price * item.quantity for item in order.items])

        message = (
            f"Generate a quote for the following order. "
            f"The order total without any discount is: {order_total}. "
            f"### Order Details: {order.model_dump_json()}\n"
            f"### Original Customer Request: {customer_request}."
        )

        try:
            response = self.agent.run_sync(message)
            return response.output
        except Exception as e:
            # Handle exceptions that may occur during quote generation
            log(f"Error generating quote: {e}", LogLevel.ERROR)
            return None

    @classmethod
    def get_system_prompt(cls) -> str:
        return """

            You are a quoting agent working in the sales department of a paper company. Your company prides itself on providing competitive pricing and excellent customer service, and it offers various discount policies to its customers to make them feel valued and appreciated as business partners.

            Your task is to generate a quote based on the order details provided by the customer. You will receive an Order object containing the items and quantities requested. 

            Instructions:
            1. Search for similar quotes in the database of past quotes to determine the best pricing / discount strategy. Use the original customer request to search for similar quotes.
            2. Apply any applicable discount policies to the quote.
            3. Calculate the total amount for the quote, including any discounts applied.
            4. Ensure that the total amount is rounded to a friendly dollar value (integer, no fraction) to make it easier for the customer to budget.
            5. Ensure that the discount is applied on the individual item level, not on the total amount to make sure that the total amount matches the sum of the individual item prices.
            6. Generate a customer quote text that summarizes the order and pricing details. Use a positive tone and highlight any discounts applied. You can search for past quotes to determine a suitable format and content for the quote text.

            It is important that the discount are applied correctly to the individual items in the quote to match the total discounted amount shown to the customer.

            Example of customer quotes:
            1. "Thank you for your large order! We have calculated the costs for 500 reams of A4 paper at $0.05 each, 300 reams of letter-sized paper at $0.06 each, and 200 reams of cardstock at $0.15 each. To reward your bulk order, we are pleased to offer a 10% discount on the total. This brings your total to a rounded and friendly price, making it easier for your budgeting needs."
            2. "For your order of 10 reams of standard copy paper, 5 reams of cardstock, and 3 boxes of assorted colored paper, I have applied a friendly bulk discount to help you save on this essential supply for your upcoming meeting. The standard pricing totals to $64.00, but with the bulk order discount, I've rounded the total cost to a more budget-friendly $60.00. This way, you receive quality materials without feeling nickel and dimed."

            Your response should include the following fields:
            - `order`: An Order object containing the order details.
            - `request_date`: The date of the request in ISO format (YYYY-MM-DD).
            - `quote_items`: A list of QuoteItem objects, each containing:
                - `item_name`: The name of the item in the quote.
                - `quantity`: The number of units quoted.
                - `discounted_price`: The price per unit of the item with discount applied.
            - `customer_quote`: The quote text provided to the customer.
            - `total_amount`: The total amount of the quote as full dollar value.
            - `discounted_total_amount`: The total amount or the order after applying discounts, if any.
            - `discounted_amount`: The total amount of discounts applied to the quote.
            - `discount_policy`: An optional DiscountPolicy object containing the discount policy applied to the quote.
        """


class TransactionAgent:
    """
    Transaction Agent responsible for managing financial transactions related to orders and inventory.

    This agent can create transactions, retrieve cash balances, and generate financial reports.
    """

    def __init__(self, model_provider: OpenAIProvider):
        self.agent_id = "transaction_agent"
        self.agent_name = "Transaction Agent"
        # This implementation does not use an LLM-based agent for transaction processing.
        # self.agent = Agent(
        #     model=OpenAIModel("gpt-4o", provider=model_provider),
        #     output_type=TransactionResult,
        #     system_prompt=self._get_system_prompt(),
        #     tools=[create_sales_transaction_tool, create_stock_order_transaction_tool],
        # )

    def process_transactions(
        self, order: Order, quote: Quote, stock_order: StockOrder
    ) -> TransactionResult:
        """
        Processes financial transactions related to an order.

        NOTE: Handling transactions for StockOrders and Sales (Quotes) does not require
        an agent call. Therefore, this method implements the logic directly.

        Args:
            order (Order): The order details for which to process transactions.
            stock_order (StockOrder): The stock order details for inventory management.

        Returns:
            TransactionResult: The result of the transaction processing, including transaction IDs.

        Raises:
            Exception: If an error occurs during transaction processing, an exception is raised.
        """
        log("Agent::process_transactions", LogLevel.INFO)

        order_request_date = order.request_date

        # First, create stock order transactions
        stock_order_transaction_ids = []
        try:

            for stock_item in stock_order.items:
                id = create_transaction(
                    item_name=stock_item.item_name,
                    transaction_type="stock_orders",
                    quantity=stock_item.quantity,
                    price=stock_item.unit_price * stock_item.quantity,
                    date=order_request_date,
                )
                stock_order_transaction_ids.append(id)

        except Exception as e:
            # Handle exceptions that may occur during transaction creation
            log(f"Error processing stock order transactions: {e}", LogLevel.ERROR)
            # IMPROVEMENT: implement compensating action that rolls the transaction
            # - rollback stock order transactions that were created until the error occurred
            raise Exception(
                "An error occurred while processing stock order transactions."
            )

        log(
            f"Stock order transactions created: {len(stock_order_transaction_ids)}",
            LogLevel.INFO,
        )

        # Second, create sales transactions based on the quote
        sales_transaction_ids = []
        try:

            for quote_item in quote.quote_items:

                id = create_transaction(
                    item_name=quote_item.item_name,
                    transaction_type="sales",
                    quantity=quote_item.quantity,
                    price=quote_item.discounted_price * quote_item.quantity,
                    date=order_request_date,
                )
                sales_transaction_ids.append(id)

        except Exception as e:
            # Handle exceptions that may occur during transaction creation
            log(f"Error processing sales transactions: {e}", LogLevel.ERROR)
            # IMPROVEMENT: implement compensating action that rolls the transaction
            # - rollback all stock order transactions
            # - rollback sales transactions that were created until the error occurred

            raise Exception("An error occurred while processing sales transactions.")

        log(f"Sales transactions created: {len(sales_transaction_ids)}", LogLevel.INFO)

        result = TransactionResult(
            sale_transaction_ids=sales_transaction_ids,
            stock_order_transaction_ids=stock_order_transaction_ids,
        )
        return result


class WorkflowError(Enum):
    ORDER_ERROR = "Failed to extract order details"
    INVENTORY_ERROR = "Inventory check failed"
    QUOTING_ERROR = "Quote generation failed"
    TRANSACTION_ERROR = "Transaction processing failed"


class OrchestrationAgent:
    """
    Orchestration Agent responsible for managing the flow of operations between other agents.

    This agent coordinates the interactions between Order, Inventory, Quoting, and Transaction agents.
    """

    def __init__(self, model_provider: OpenAIProvider):
        self.agent_id = "orchestration_agent"
        self.agent_name = "Orchestration Agent"
        self.order_agent = OrderAgent(model_provider=model_provider)
        self.inventory_agent = InventoryAgent(model_provider=model_provider)
        self.quoting_agent = QuotingAgent(model_provider=model_provider)
        self.transaction_agent = TransactionAgent(model_provider=model_provider)

    def process_quote_request(self, request_text: str) -> str:
        """
        Represents the main entry point for processing a customer quote request.

        This method implements the orchestration logic to manage the flow between agents,
        extracting order details, checking inventory, generating quotes, and processing transactions.

        Args:
            request_text (str): The text of the customer request containing order information.

        Returns:
            OrderAgentResponse: Contains the extracted order details or an error message.
        """

        # Process memory: this represents the main memory of the process.
        # This can later be transformed into a proper (persistent) memory management.
        order = None
        stock_order = None
        quote = None

        # Step 1: Extract order details using the Order Agent
        with status("Order Agent - process_quote_request") as s:
            order_response = self.order_agent.process_quote_request(
                request_text=request_text
            )
            if not order_response.is_success or not order_response.order:
                s.fail("Order extraction failed")
                return self._handle_error(
                    WorkflowError.ORDER_ERROR, agent_error=order_response.agent_error
                )

        # update process memory: order
        order = order_response.order

        with status("Inventory Agent - process_order") as s:
            # Step 2: Check inventory using the Inventory Agent
            # inventory_res = self.inventory_agent.process_order_direct(order=order)
            inventory_res = self.inventory_agent.process_order_llm(order=order)

            if not inventory_res or not inventory_res.is_success:
                s.fail("Inventory check failed")
                return self._handle_error(
                    WorkflowError.INVENTORY_ERROR, agent_error=inventory_res.agent_error
                )

        # update process memory: stock order
        stock_order = inventory_res.stock_order

        # Step 3: Generate quote using the Quoting Agent
        with status("Quoting Agent - generate_quote") as s:
            quote_res = self.quoting_agent.generate_quote(
                order=order, customer_request=request_text
            )
            if not quote_res:
                s.fail("Quote generation failed")
                return self._handle_error(
                    WorkflowError.QUOTING_ERROR,
                    "We cannot provide a quote for your request at this point in time.",
                )

        log(f"Quote generated:", LogLevel.DEBUG)
        log(quote_res.model_dump_json(indent=2), LogLevel.DEBUG)

        # update process memory: quote
        quote = quote_res

        # Step 4: Process payment using the Transaction Agent
        with status("Transaction Agent - process_transactions") as s:
            # Process transactions for the order and quote
            transaction_response = self.transaction_agent.process_transactions(
                order=order, quote=quote, stock_order=stock_order
            )
            if not transaction_response:
                return self._handle_error(
                    WorkflowError.TRANSACTION_ERROR,
                    "An error occurred while processing transactions.",
                )

        # return the order response to the caller
        response = QuoteResult(order=order, quote=quote)
        result_str = str(response)

        return result_str

    def _handle_error(self, error: WorkflowError, agent_error: str) -> str:
        """
        Handles errors that occur during the orchestration process.

        Args:
            error (WorkflowError): The error that occurred.

        Returns:
            OrderAgentResponse: An error response containing the error message.
        """
        return f"Ordering Error: {agent_error}"


"""Set up tools for your agents to use, these should be methods that combine the database functions above
 and apply criteria to them to ensure that the flow of the system is correct."""


def info_tool_call(func):
    """
    Decorator that prints the function name and returns the function with its doc-string.
    Intended to be used with tool decorators for debugging and documentation.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        log(f"Tool: {func.__name__}", LogLevel.INFO)
        return func(*args, **kwargs)

    return wrapper


def tool(func):
    """
    Decorator to mark a function as a Pydantic-AI Tool to be used by an agents.
    """
    return Tool(info_tool_call(func))


def get_all_inventory_items() -> List[InventoryItem]:
    """
    Retrieves all items from the inventory as of the current date.

    Returns:
        List[InventoryItem]: A list of InventoryItem objects representing all items in the inventory.

    """
    # Query the inventory table to get all items
    inventory_df = pd.read_sql("SELECT * FROM inventory", db_engine)

    # Convert DataFrame rows to InventoryItem objects
    items = [
        InventoryItem(
            item_name=row["item_name"],
            category=row["category"],
            unit_price=row["unit_price"],
            min_stock_level=row["min_stock_level"],
        )
        for _, row in inventory_df.iterrows()
    ]

    return items


@tool
def get_all_inventory_items_tool() -> List[InventoryItem]:
    """
    Retrieves all items from the inventory as of the current date.

    Returns:
        List[InventoryItem]: A list of InventoryItem objects representing all items in the inventory.
    """
    # Call the get_all_inventory_items function to retrieve all items
    return get_all_inventory_items()


def get_inventory_item(item_name: str) -> InventoryItem:
    """
    Retrieves a specific item from the inventory by its name.

    Args:
        item_name (str): The name of the item to retrieve.

    Returns:
        InventoryItem: The InventoryItem object representing the requested item. Or None if the item is not found.
    """
    # Query the inventory table to get the item by name
    query = f"SELECT * FROM inventory WHERE item_name = '{item_name}'"
    inventory_df = pd.read_sql(query, db_engine)

    if inventory_df.empty:
        return None

    row = inventory_df.iloc[0]
    return InventoryItem(
        item_name=row["item_name"],
        category=row["category"],
        unit_price=row["unit_price"],
        min_stock_level=row["min_stock_level"],
    )


@tool
def get_inventory_item_tool(item_name: str) -> InventoryItem:
    """
    Retrieves a specific item from the inventory by its name.

    Args:
        item_name (str): The name of the item to retrieve.

    Returns:
        InventoryItem: The InventoryItem object representing the requested item. Or None if the item is not found.
    """
    # Call the get_inventory_item function to retrieve the item
    return get_inventory_item(item_name=item_name)


@tool
def get_order_id_tool(request_text: str) -> str:
    """
    Extracts the order ID from a customer request text.

    Args:
        request_text (str): The text of the customer request containing order information.

    Returns:
        str: The extracted order ID.
    """

    # For now, we assume that each call generates a new order ID.
    return uuid.uuid4().hex


@tool
def search_quote_history_tool(order: Order, customer_request: str) -> DiscountPolicy:
    """
    Returns a list of similar quotes from the quote history based on the customer request.

    Args:
        order (Order): The order details for which to retrieve the discount policy.

    Returns:
        List[Dict]: A list of matching quotes, each represented as a dictionary with fields:
            - original_request
            - total_amount
            - quote_explanation
            - job_type
            - order_size
            - event_type
            - order_date
    """
    log("Searching Quote History", LogLevel.DEBUG)
    log(f"Original Customer Request: {customer_request}", LogLevel.DEBUG)

    # run a search for quotes that match the order items
    search_terms = [item.item_name.strip().lower() for item in order.items]
    search_results = search_quote_history(search_terms=search_terms, limit=5)
    if search_results and len(search_results) > 0:
        log(
            f"🔍 Found similar quotes: {len(search_results)} for products {', '.join(search_terms)}",
            LogLevel.INFO,
        )

    log(
        f"Search results for quote history found: {len(search_results)}", LogLevel.DEBUG
    )
    for result in search_results:
        log(
            f"- {result['original_request']} \n- {result['quote_explanation']}",
            LogLevel.DEBUG,
        )
    return search_results


@tool
def get_cash_balance_tool(date: str) -> float:
    """
    Retrieves the cash balance as of a specific date.

    Args:
        date (str): The date in ISO format (YYYY-MM-DD) for which to retrieve the cash balance.

    Returns:
        float: The cash balance as of the specified date.
    """
    # Call the get_cash_balance function to retrieve the cash balance
    return get_cash_balance(date=date)


@tool
def get_supplier_delivery_date_tool(order_date: str, quantity: int) -> str:
    """
    Retrieves the expected delivery date from the supplier for a given order date and quantity.

    Args:
        order_date (str): The order date in ISO format (YYYY-MM-DD).
        quantity (int): The quantity of items ordered.

    Returns:
        str: The expected delivery date in ISO format (YYYY-MM-DD).
    """
    # Call the get_supplier_delivery_date function to retrieve the delivery date
    return get_supplier_delivery_date(order_date=order_date, quantity=quantity)


@tool
def get_inventory_item_tool(item_name: str) -> InventoryItem:
    """
    Retrieves a specific item from the inventory by its name.

    Args:
        item_name (str): The name of the item to retrieve.

    Returns:
        InventoryItem: The InventoryItem object representing the requested item. Or None if the item is not found.
    """
    # Call the get_inventory_item function to retrieve the item
    return get_inventory_item(item_name=item_name)


@tool
def get_stock_level_tool(item_name: str, as_of_date: str) -> pd.DataFrame:
    """
    Retrieves the stock level of a specific item as of a given date.

    Args:
        item_name (str): The name of the item to check.
        as_of_date (str): The date in ISO format (YYYY-MM-DD) to check stock levels against.

    Returns:
        pd.DataFrame: A DataFrame containing the stock level data for the specified item.
    """
    # Call the get_stock_level function to retrieve the stock level
    return get_stock_level(item_name=item_name, as_of_date=as_of_date)


# Set up your agents and create an orchestration agent that will manage them.


# Run your test scenarios by writing them here. Make sure to keep track of them.


# load environment
dotenv.load_dotenv()
UDACITY_OPENAI_API_KEY = os.getenv("UDACITY_OPENAI_API_KEY")

log_level = LogLevel.WARNING


def run_test_scenarios():

    log("Initializing Database...", LogLevel.INFO)
    init_database(db_engine=db_engine)
    try:
        quote_requests_sample = pd.read_csv("quote_requests_sample.csv")
        quote_requests_sample["request_date"] = pd.to_datetime(
            quote_requests_sample["request_date"], format="%m/%d/%y", errors="coerce"
        )
        quote_requests_sample.dropna(subset=["request_date"], inplace=True)
        quote_requests_sample = quote_requests_sample.sort_values("request_date")
    except Exception as e:
        log(f"FATAL: Error loading test data: {e}", LogLevel.ERROR)
        return

    quote_requests_sample = pd.read_csv("quote_requests_sample.csv")

    # Sort by date
    quote_requests_sample["request_date"] = pd.to_datetime(
        quote_requests_sample["request_date"]
    )
    quote_requests_sample = quote_requests_sample.sort_values("request_date")

    # Get initial state
    initial_date = quote_requests_sample["request_date"].min().strftime("%Y-%m-%d")
    report = generate_financial_report(initial_date)
    current_cash = report["cash_balance"]
    current_inventory = report["inventory_value"]

    ############
    ############
    ############
    # INITIALIZE YOUR MULTI AGENT SYSTEM HERE
    ############
    ############
    ############
    # Set up OpenAI provider with the API key
    openai_provider = OpenAIProvider(
        base_url="https://openai.vocareum.com/v1", api_key=UDACITY_OPENAI_API_KEY
    )

    orchestration_agent = OrchestrationAgent(model_provider=openai_provider)

    # define how many sample to process: defaults to all
    # sample_limit = 5  # Set to a smaller number for testing purposes
    sample_limit = len(quote_requests_sample)

    results = []
    for idx, row in quote_requests_sample.iterrows():
        request_date = row["request_date"].strftime("%Y-%m-%d")

        print(f"\n=== Request {idx+1} ===")
        print(f"Context: {row['job']} organizing {row['event']}")
        print(f"Request Date: {request_date}")
        print(f"Cash Balance: ${current_cash:.2f}")
        print(f"Inventory Value: ${current_inventory:.2f}")

        # Process request
        request_with_date = f"{row['request']} (Date of request: {request_date})"

        ############
        ############
        ############
        # USE YOUR MULTI AGENT SYSTEM TO HANDLE THE REQUEST
        ############
        ############
        ############
        response = orchestration_agent.process_quote_request(
            request_text=request_with_date
        )

        # Update state
        report = generate_financial_report(request_date)
        current_cash = report["cash_balance"]
        current_inventory = report["inventory_value"]

        print(f"Response: {response}")
        print(f"Updated Cash: ${current_cash:.2f}")
        print(f"Updated Inventory: ${current_inventory:.2f}")

        results.append(
            {
                "request_id": idx + 1,
                "request_date": request_date,
                "cash_balance": current_cash,
                "inventory_value": current_inventory,
                "response": response,
            }
        )

        time.sleep(1)

        if idx >= sample_limit - 1:
            break

    # Final report
    final_date = quote_requests_sample["request_date"].max().strftime("%Y-%m-%d")
    final_report = generate_financial_report(final_date)
    print("\n===== FINAL FINANCIAL REPORT =====")
    print(f"Final Cash: ${final_report['cash_balance']:.2f}")
    print(f"Final Inventory: ${final_report['inventory_value']:.2f}")

    # Save results
    pd.DataFrame(results).to_csv("test_results.csv", index=False)
    return results


def run_test_scenario_by_index(idx: int):

    log("Initializing Database...", LogLevel.INFO)
    init_database(db_engine=db_engine)
    try:
        quote_requests_sample = pd.read_csv("quote_requests_sample.csv")
        quote_requests_sample["request_date"] = pd.to_datetime(
            quote_requests_sample["request_date"], format="%m/%d/%y", errors="coerce"
        )
        quote_requests_sample.dropna(subset=["request_date"], inplace=True)
        quote_requests_sample = quote_requests_sample.sort_values("request_date")
    except Exception as e:
        log(f"FATAL: Error loading test data: {e}", LogLevel.ERROR)
        return

    quote_requests_sample = pd.read_csv("quote_requests_sample.csv")

    # Sort by date
    quote_requests_sample["request_date"] = pd.to_datetime(
        quote_requests_sample["request_date"]
    )
    quote_requests_sample = quote_requests_sample.sort_values("request_date")

    # Get initial state
    initial_date = quote_requests_sample["request_date"].min().strftime("%Y-%m-%d")
    report = generate_financial_report(initial_date)
    current_cash = report["cash_balance"]
    current_inventory = report["inventory_value"]

    # Set up OpenAI provider with the API key
    openai_provider = OpenAIProvider(
        base_url="https://openai.vocareum.com/v1", api_key=UDACITY_OPENAI_API_KEY
    )

    orchestration_agent = OrchestrationAgent(model_provider=openai_provider)

    results = []
    row = quote_requests_sample.iloc[idx]

    request_date = row["request_date"].strftime("%Y-%m-%d")

    print(f"\n=== Request {idx+1} ===")
    print(f"Context: {row['job']} organizing {row['event']}")
    print(f"Request Date: {request_date}")
    print(f"Cash Balance: ${current_cash:.2f}")
    print(f"Inventory Value: ${current_inventory:.2f}")

    # Process request
    request_with_date = f"{row['request']} (Date of request: {request_date})"

    response = orchestration_agent.process_quote_request(request_text=request_with_date)

    # Update state
    report = generate_financial_report(request_date)
    current_cash = report["cash_balance"]
    current_inventory = report["inventory_value"]

    print(f"Response: {response}")
    print(f"Updated Cash: ${current_cash:.2f}")
    print(f"Updated Inventory: ${current_inventory:.2f}")

    results.append(
        {
            "request_id": idx + 1,
            "request_date": request_date,
            "cash_balance": current_cash,
            "inventory_value": current_inventory,
            "response": response,
        }
    )

    # Final report
    final_date = quote_requests_sample["request_date"].max().strftime("%Y-%m-%d")
    final_report = generate_financial_report(final_date)
    print("\n===== FINAL FINANCIAL REPORT =====")
    print(f"Final Cash: ${final_report['cash_balance']:.2f}")
    print(f"Final Inventory: ${final_report['inventory_value']:.2f}")

    # Save results
    pd.DataFrame(results).to_csv("test_results.csv", index=False)
    return results


if __name__ == "__main__":
    # results = run_test_scenarios()
    results = run_test_scenario_by_index(0)  # Run a specific test scenario by index
