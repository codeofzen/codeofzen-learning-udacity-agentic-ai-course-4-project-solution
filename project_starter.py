from contextlib import contextmanager
from dataclasses import dataclass
import json
from queue import Empty, Queue
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
        inventory_df = generate_sample_inventory(paper_supplies, seed=seed)

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
        return [dict(row) for row in result]


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


log_level = LogLevel.DEBUG


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

    def fail(self, reason: str = "Operation failed"):
        """Manually mark the task as failed"""
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
    Represents an item in the inventory with its name, category, unit price,
    current stock, and minimum stock level.
    """

    item_name: str = Field(..., description="Name of the inventory item")
    category: str = Field(..., description="Category of the inventory item")
    unit_price: float = Field(..., description="Price per unit of the inventory item")


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
    items: List[OrderItem] = Field(..., description="List of items in the order")


class OrderResponse(BaseModel):
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
    expected_delivery_date: str = Field(
        ...,
        description="Expected delivery date for the stock order in ISO format (YYYY-MM-DD)",
    )


class StockOrder(BaseModel):
    """
    Represents a stock order for inventory management.
    """

    items: List[StockOrderItem] = Field(
        ..., description="List of items in the stock order"
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
    discounted_amount: float = Field(..., description="Discounted amount of the quote")
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


class Response(BaseModel):
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

        return f"{self.quote.discounted_amount}, {self.quote.customer_quote}, {json.dumps(order_details)}"


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
            output_type=OrderResponse,
            system_prompt=self._get_system_prompt(),
            tools=[get_order_id_tool, get_items_from_inventory],
        )

    def process_quote_request(self, request_text: str) -> OrderResponse:
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
                    "Order extraction failed: %s",
                    order_output.agent_error,
                    LogLevel.ERROR,
                )
        except Exception as e:
            log("Error processing quote requests", LogLevel.ERROR)
            order_output = OrderResponse(
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
        # TODO: Improve system prompt
        return """
            You are an sales agent working in the sales department of a paper company.

            Your task is to process incoming order requests from customers and extract the relevant order details from the request.  You will receive a customer request as input, which may contain various details about the order, such as item names, quantities, and any special instructions. Ensure that the items match the items available in the inventory matching their names.

            Your response should include the following fields:
            - `is_success`: A boolean indicating whether the order extraction was successful.
            - `order`: An Order object containing the extracted order details, including:
                - `items`: A list of OrderItem objects, each containing:
                    - `item_name`: The name of the item being ordered.
                    - `quantity`: The number of units to order.
                    - `unit_price`: The price per unit of the item being ordered.

            Provide quality always as units (paper sheets, etc.). This means that you need to convert any other units (e.g., reams, packs) into units.
            - `reams`: A ream is 500 units.
            - `pack`: A pack is 100 units.
            - `box`: A box is 5000 units.
            
            If the request does not contain sufficient information to extract an order, set `is_success` to False and provide an appropriate error message in the `agent_error` field.
            
            Example: 
            {
                "is_success": False,
                "agent_error": "The request does not contain sufficient information to extract an order."
            }    
            """


class InventoryAgent:
    """
    Inventory Agent responsible for managing and querying inventory data.

    This agent can check stock levels, reorder items, and provide inventory reports.
    """

    def __init__(self, model_provider: OpenAIProvider):
        self.agent_id = "inventory_agent"
        self.agent_name = "Inventory Agent"
        # This implementation does not use an LLM-based agent for inventory management.
        # self.agent = Agent(
        #     model=OpenAIModel("gpt-4o", provider=model_provider),
        #     output_type=StockOrder,
        #     system_prompt=self._get_system_prompt(),
        # )

    def process_order(self, order: Order) -> StockOrder:
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
        try:

            res = process_order(order=order)
            return res
        except Exception as e:
            # Handle exceptions that may occur during order processing
            log(f"Error processing order: {e}", LogLevel.ERROR)
            return None

    # Not used in this implementation
    def _get_system_prompt(self) -> str:
        """
        Returns the system prompt for the Inventory Agent.
        This prompt guides the agent's behavior and expectations.
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
            4. Ensure that the discount is applied on the individual item level, not on the total amount to make sure that the total amount matches the sum of the individual item prices.
            5. Generate a customer quote text that summarizes the order and pricing details. Use a positive tone and highlight any discounts applied. You can search for past quotes to determine a suitable format and content for the quote text.

            It is important that the discount are applied correctly to the individual items in the quote to match the total discounted amount shown to the customer.

            Your response should include the following fields:
            - `order`: An Order object containing the order details.
            - `quote_items`: A list of QuoteItem objects, each containing:
                - `item_name`: The name of the item in the quote.
                - `quantity`: The number of units quoted.
                - `discounted_price`: The price per unit of the item with discount applied.
            - `customer_quote`: The quote text provided to the customer.
            - `total_amount`: The total amount of the quote.
            - `discounted_amount`: The discounted amount of the quote.
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
        self, quote: Quote, stock_order: StockOrder
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
        stock_order_transaction_ids = []

        try:

            for stock_item in stock_order.items:
                id = create_transaction(
                    item_name=stock_item.item_name,
                    transaction_type="stock_orders",
                    quantity=stock_item.quantity,
                    price=stock_item.unit_price * stock_item.quantity,
                    date=stock_item.date,
                )
                stock_order_transaction_ids.append(id)

        except Exception as e:
            # Handle exceptions that may occur during transaction creation
            log(f"Error processing stock order transactions: {e}", LogLevel.ERROR)
            # TODO: implement compensating action that rolls the transaction
            # - rollback stock order transactions that were created until the error occurred
            raise Exception(
                "An error occurred while processing stock order transactions."
            )

        log(
            f"Stock order transactions created: {len(stock_order_transaction_ids)}",
            LogLevel.INFO,
        )

        sales_transaction_ids = []

        try:

            for quote_item in quote.quote_items:

                id = create_transaction(
                    item_name=quote_item.item_name,
                    transaction_type="sales",
                    quantity=quote_item.quantity,
                    price=quote_item.discounted_price * quote_item.quantity,
                    date=datetime.now(),
                )
                sales_transaction_ids.append(id)

        except Exception as e:
            # Handle exceptions that may occur during transaction creation
            log(f"Error processing sales transactions: {e}", LogLevel.ERROR)
            # TODO: implement compensating action that rolls the transaction
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

        # Step 1: Extract order details using the Order Agent
        with status("Order Agent - process_quote_request") as s:
            order_response = self.order_agent.process_quote_request(
                request_text=request_text
            )
            if not order_response.is_success or not order_response.order:
                s.fail("Order extraction failed")
                return self._handle_error(WorkflowError.ORDER_ERROR)

            s.fail("Order extraction successful")

        order = order_response.order

        # Step 2: Check inventory using the Inventory Agent
        stock_order = self.inventory_agent.process_order(order=order)
        if not stock_order:
            return self._handle_error(WorkflowError.INVENTORY_ERROR)

        # Step 3: Generate quote using the Quoting Agent
        quote = self.quoting_agent.generate_quote(
            order=order, customer_request=request_text
        )
        if not quote:
            return self._handle_error(WorkflowError.QUOTING_ERROR)

        log(f"Quote generated:", LogLevel.DEBUG)
        log(quote.model_dump_json(indent=2), LogLevel.DEBUG)

        # Step 4: Process payment using the Transaction Agent
        transaction_response = self.transaction_agent.process_transactions(
            quote=quote, stock_order=stock_order
        )
        if not transaction_response:
            return self._handle_error(WorkflowError.TRANSACTION_ERROR)

        # return the order response to the caller
        response = Response(order=order, quote=quote)
        return str(response)

    def _handle_error(self, error: WorkflowError) -> str:
        """
        Handles errors that occur during the orchestration process.

        Args:
            error (WorkflowError): The error that occurred.

        Returns:
            OrderAgentResponse: An error response containing the error message.
        """

        if error == WorkflowError.ORDER_ERROR:
            log(
                "[ERROR] Workflow Error: Failed to extract order details.",
                LogLevel.ERROR,
            )
            return "We encountered an issue while extracting order details."
        else:
            return f"We encountered a problem while processing your request."


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


@tool
def get_items_from_inventory() -> List[InventoryItem]:
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
        )
        for _, row in inventory_df.iterrows()
    ]

    return items


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


# Tools for inventory agent
def process_order(order: Order) -> StockOrder:
    """
    Processes an order by checking stock levels and returning a stock order if necessary.

    Args:
        order (Order): The order details to process.

    Returns:
        StockOrder: A StockOrder object containing the item name, quantity, and expected delivery date.
    """
    # loop through all order items and check the stock level
    stock_orders = []
    for item in order.items:
        # query stock level
        stock_level = get_stock_level(item.item_name, datetime.now())

        # check if item was found in the inventory
        if stock_level.empty:
            log(f"Item {item.item_name} not found in inventory.", LogLevel.ERROR)
            # Error Strategy: be opportunistic and skip the unknown item. This can be improve at a later stage.
            # A better strategy would be to direct the request to a human agent or to log the error for further investigation.
            continue

        if stock_level["current_stock"].iloc[0] < item.quantity:
            # If stock is below the required quantity, create a stock order
            expected_delivery_date = get_supplier_delivery_date(
                datetime.now().isoformat(), item.quantity
            )
            stock_orders.append(
                StockOrderItem(
                    item_name=item.item_name,
                    quantity=item.quantity,
                    expected_delivery_date=expected_delivery_date,
                )
            )

    # return a stock order with empty items if no stock orders were created
    return StockOrder(items=stock_orders)


@tool
def process_order_tool(order: Order) -> StockOrder:
    """
    Processes an order by checking stock levels and returning a stock order if necessary.

    Args:
        order (Order): The order details to process.

    Returns:
        StockOrder: A StockOrder object containing the item name, quantity, and expected delivery date.
    """
    # Call the process_order function to check stock levels and create stock orders
    return process_order(order=order)


def calculate_order_total(order: Order) -> float:
    """
    Calculates the total amount for a given order.

    Args:
        order (Order): The order details for which to calculate the total amount.

    Returns:
        float: The total amount for the order.
    """
    total_amount = sum(item.unit_price * item.quantity for item in order.items)
    return total_amount


@tool
def calculate_order_total_tool(order: Order) -> float:
    """
    Calculates the total amount for a given order.

    Args:
        order (Order): The order details for which to calculate the total amount.
        customer_request (str): The original customer request for context.

    Returns:
        float: The total amount for the order.
    """
    # Call the calculate_order_total function to compute the total amount
    return calculate_order_total(order=order)


# @QuotingAgent.tool
@tool
def search_quote_history_tool(order: Order, customer_request: str) -> DiscountPolicy:
    """
    Retrieves the discount policy applicable to a given order.

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
    # search_results = search_quote_history(
    #     search_terms=[item.item_name for item in order.items], limit=5
    # )
    search_results = search_quote_history(search_terms=[customer_request], limit=5)

    log(f"Search results for quote history found: {len(search_results)}", LogLevel.INFO)
    for result in search_results:
        log(
            f"- {result['original_request']} \n- {result['quote_explanation']}",
            LogLevel.DEBUG,
        )
    return search_results


# Set up your agents and create an orchestration agent that will manage them.


# Run your test scenarios by writing them here. Make sure to keep track of them.


# load environment
dotenv.load_dotenv()
UDACITY_OPENAI_API_KEY = os.getenv("UDACITY_OPENAI_API_KEY")


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
        response = orchestration_agent.process_quote_request(request_with_date)

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

        # TODO REMOVE this after testing
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


if __name__ == "__main__":
    results = run_test_scenarios()
