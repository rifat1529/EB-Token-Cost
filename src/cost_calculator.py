"""Course-notional token pricing."""
INPUT_PRICE_PER_MILLION = 0.15
OUTPUT_PRICE_PER_MILLION = 0.60


def calculate_cost(input_tokens: int, output_tokens: int) -> dict[str, float]:
    if input_tokens < 0 or output_tokens < 0:
        raise ValueError("Token counts cannot be negative.")

    input_cost = input_tokens * INPUT_PRICE_PER_MILLION / 1_000_000
    output_cost = output_tokens * OUTPUT_PRICE_PER_MILLION / 1_000_000
    return {
        "input_cost_usd": input_cost,
        "output_cost_usd": output_cost,
        "cost_usd": input_cost + output_cost,
    }
