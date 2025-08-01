# Project Notes

These notes are internal notes that capture specific question or ideas during the design and development stages.

### Evaluations

Context from the instructions:

> Ensure:
>
> -   Your agents correctly handle various customer inquiries and orders
> -   Orders are accommodated effectively to optimize inventory use and profitability
> -   The quoting agent consistently provides competitive and attractive pricing

### Customer Agent

> Context: Create a customer agent that uses the customer context from the sample requests to then can negotiate with the team multi-agent system here.

-   Provide the agent with a goal to optimize for. What is a good result for a negotiation?
-   Provide the agent with a negotiation strategy: soft, medium, hard. How is this related to the goal?
-   Provide the agent with specific degrees of freedom to accept proposals from the QuoteManagement Agent. What compromises are possible to get a better deal for both sides?

### Business Advisor Agent

> Context: Add a business advisor agent that analyses all the transactions being handled by the multi-agent system and proactively recommends changes to the business operations in order to improve its efficiency and revenue.

1. Increase Revenue: increase order volume, reduce discounts, delay stock re-order until data closer to fulfillment (less inventory, later payment)
2. Improve Efficiency:

Assumptions:

-   Order Shipment Costs: each individual order includes additional charges for shipment
-   Inventory Operations: Less stock orders reduce work in the warehouse and logistics

#### Objectives 1: Increase Revenue

TODO

#### Objective 2: Improve Efficiancy

TODO
