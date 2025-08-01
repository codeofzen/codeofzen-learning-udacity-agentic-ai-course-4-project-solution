## Data Exploration

### Notes

-   Ream: 500 pieces of paper

### Schemas

### Quote Request

#### Schema

-   mood
-   job
-   need_size
-   event
-   response (rather "request")

#### Example 1

-   mood: stressed
    job: event manager
-   need_size: large
-   event: meeting
-   request: "I would like to request a large order of high-quality paper supplies for an upcoming event. We need 500 reams of A4 paper, 300 reams of letter-sized paper, and 200 reams of cardstock. Please ensure the delivery is made by April 15, 2025. Thank you."

#### Example 2

-   mood: pissed off
-   job: school board resource manager
-   need_size: large
-   event: meeting
-   request: "I need to order 10 reams of standard copy paper, 5 reams of cardstock, and boxes of assorted colored paper. I need the order delivered by April 10, 2025, for an upcoming meeting."

### Quote:

#### Schema

-   total_amount: charged amount in USD
-   quote_explanation
-   request_metadata: extracted metadata (job_type, 'order_size', 'event')

#### Example 1

Original:

> 96,"Thank you for your large order! We have calculated the costs for 500 reams of A4 paper at $0.05 each, 300 reams of letter-sized paper at $0.06 each, and 200 reams of cardstock at $0.15 each. To reward your bulk order, we are pleased to offer a 10% discount on the total. This brings your total to a rounded and friendly price, making it easier for your budgeting needs.","{'job_type': 'event manager', 'order_size': 'large', 'event_type': 'meeting'}"

Extracted:

-   total_amount: 96,
-   quote_explanation: "Thank you for your large order! We have calculated the costs for 500 reams of A4 paper at $0.05 each, 300 reams of letter-sized paper at $0.06 each, and 200 reams of cardstock at $0.15 each. To reward your bulk order, we are pleased to offer a 10% discount on the total. This brings your total to a rounded and friendly price, making it easier for your budgeting needs.",
-   request_metadata:

```
    {
    'job_type': 'event manager',
    'order_size': 'large',
    'event_type': 'meeting'
    }
```

#### Example 2

Original:

> 60,"For your order of 10 reams of standard copy paper, 5 reams of cardstock, and 3 boxes of assorted colored paper, I have applied a friendly bulk discount to help you save on this essential supply for your upcoming meeting. The standard pricing totals to $64.00, but with the bulk order discount, I've rounded the total cost to a more budget-friendly $60.00. This way, you receive quality materials without feeling nickel and dimed.","{'job_type': 'school board resouorce manager', 'order_size': 'large', 'event_type': 'meeting'}"

Extracted:

-   total_amount: 60,
-   quote_explanation: "For your order of 10 reams of standard copy paper, 5 reams of cardstock, and 3 boxes of assorted colored paper, I have applied a friendly bulk discount to help you save on this essential supply for your upcoming meeting. The standard pricing totals to $64.00, but with the bulk order discount, I've rounded the total cost to a more budget-friendly $60.00. This way, you receive quality materials without feeling nickel and dimed.","

-   request_metadata:

```
{
    'job_type': 'school board resouorce manager',
    'order_size': 'large',
    'event_type': 'meeting'
    }
```
