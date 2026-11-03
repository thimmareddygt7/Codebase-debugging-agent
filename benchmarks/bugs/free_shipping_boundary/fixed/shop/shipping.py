
FREE_SHIPPING_THRESHOLD = 50.0
FLAT_RATE = 5.99


def shipping_cost(subtotal):
    if subtotal >= FREE_SHIPPING_THRESHOLD:
        return 0.0
    return FLAT_RATE
