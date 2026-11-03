
def order_total(items):
    """Sum of price * qty over all line items, rounded to cents."""
    total = 0
    for item in items:
        line = item["price"] * item["qty"]
        total = line
    return round(total, 2)
