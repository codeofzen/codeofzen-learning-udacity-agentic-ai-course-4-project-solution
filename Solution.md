# Reflections on Project Assignment

This document contains the documentation for the "Beaver's Choice Paper Company" / "Munder Difflin" Multi-Agent System project for course 4 of the Udacity Nanodegree **AI Agents**.

## 1. Project Objective

The objective is to implement a smart, adaptable system for the company to automate it's sales process with the help of AI. The process consists of the main following steps:

-   Inventory checks and restocking decisions
-   Quote generation for incoming sales inquiries
-   Order fulfillment including supplier logistics and transactions

Some assumptions had to be made by the author to come up with a solution and these assumptions have been documented in the section: [Assumptions & Notes](#assumptions--notes)

### Functional Requirements

TODO

#### Non-Functional Requirements

## 2. Architecture

TODO: Architecture Diagram

### Selected Agent Framework

Pydantic-AI has been selected as the framework for building the systems. The key criteria was the build in support for Structured Outputs (via Pydantic), which was considered a big support when building reliable agents.

### Agent Documentation

#### Orchestration Agent

The Orchestration Agent controls the dataflow between the individual agents. While the business logic implemented in this agent is minimal (mainly the dataflow logic), the agent is a critical component that provides

-   _Memory Management_: the short-term memory management that is required during the process.
-   _Error Handling_: in case errors are encountered in the process, the Orchestration Agent executes compensating actions that it tracks across the duration of the process.
-   _Process Logging_: logs each step for analysis and auditing

##### Interfaces

-   Input Schema: Message (str) including the customer request and the expected data of delivery.
-   Output Schema: CustomerQuote
-   Tools: none

#### Sales Desk Agent

The Sales Desk Agent represents the initial agent that is called to process the customer request. It therefore processes the request and extracts the relevant data for the following process steps. The agent is also responsible for handling customer messages that aren't valid quote requests. In this case, an error is returned to the Orchestration Agent.

##### Interfaces

-   Input Schema: Customer message (str)
-   Output Schema: Order or Error
-   Tools:
    -   TODO

#### Quoting Agent

The Quoting Agent specialized on generating quotes for customer orders. This includes the steps:

1. Calculate amount: based on order details
2. Determine discount policy for order:
    - Search for similar quotes
    - Identify Quoting policy: None, Round-Down, Percentage Discount
3. Generate Quote with expected information

##### Interfaces

-   Input Schema: Order
-   Output Schema: Quote
-   Tools:
    -   TODO

#### Inventory Agent

The Inventory Agent encapsulates the fulfillment of orders and the necessary inventory management. The agent returns an InventoryUpdate object that contains all items with due dates and the respective re-order amount.

We assume that the deliver from the inventory to the customer will take 2 days. Therefore, the expected delivery data for the re-orders in -2 days relative to the customer order date.

The inventory is not actively managed by the agent (adding or reducing stock levels for items) since tracking stock levels are determined via the Transaction history.

##### Interfaces

-   Input Schema: Order
-   Output Schema: InventoryUpdate
-   Tools:
    -   TODO

#### Transaction Agent

The Transaction Agents handles both Sales and Stock Order transactions.

##### Interfaces

-   Input Schema: Transaction
-   Output Schema: TransactionResult
-   Tools:
    -   TODO

### Dataflow Documentation

TODO

## 3. Improvements & Opportunities

### Parallelization

The current implementation of the process is sequential. Nevertheless, the agents "Quoting" and "Inventory" can run in parallel since both are independent.

### Updating the Inventory table

The inventory is technically managed using an event-sourcing approach, using both past Sales and StockOrder transactions to track the stock level. This help when implementing compensating actions for errors identifying the associated transactions (events).

While this is a very elegant way, it will eventually lead to performance problems. The inventory table can therefore be updated periodically. This requires that a proper timestamp is persisted to combine both data from the inventory with the

This is not directly related to the agentic architecture but rather to the general architecture of event driven systems.

## 4. Assumptions & Notes

### Paper Industry-Specific Quantity Definitions

Some of the quote-requests contain amount specified as "reams" or "boxes". The inventory and pricing on the other hand are specified as "units", which was read as "per paper". Therefore, following mapping assumptions have been made after some research:

-   ream: 500 units (papers)
-   box: 5000 units (papers)
