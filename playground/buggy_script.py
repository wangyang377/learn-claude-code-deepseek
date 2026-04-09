def calculate_total(prices: list[float], tax_rate: float) -> float:
    subtotal = sum(prices)
    total = subtotal * (1 + tax_rate)
    return round(total, 2)


def format_summary(user_name: str, total: float, item_count: int) -> str:
    return f"User {user_name} bought {item_count} items, total=${total:.2f}"


def main() -> None:
    prices = [19.9, 5.0, 3.5]
    tax_rate = 0.1
    total = calculate_total(prices, tax_rate)
    summary = format_summary("Alice", total, len(prices))
    print(summary)


if __name__ == "__main__":
    main()
