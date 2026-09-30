"""Run: python3 test_agent.py — fails loudly if the ranking logic breaks."""
from app import compare, dedupe, score_product

BUDGET = 70_000
MUST = ["rtx", "16gb"]

ITEMS = [
    {"name": "Acer Nitro V 15", "price_inr": 67_990, "key_specs": ["RTX 4050 6GB", "16GB RAM"], "source_url": "a"},
    {"name": "acer  nitro-V 15", "price_inr": 0, "key_specs": [], "source_url": "b"},          # dupe, no price
    {"name": "Lenovo IdeaPad Slim 3", "price_inr": 42_000, "key_specs": ["8GB RAM"], "source_url": "c"},
    {"name": "MacBook Pro M4", "price_inr": 169_900, "key_specs": ["24GB RAM"], "source_url": "d"},
]


def main():
    assert len(dedupe(ITEMS)) == 3, "name-normalised dedupe failed"
    assert dedupe(ITEMS)[0]["price_inr"] == 67_990, "dedupe must keep the priced row"

    ranked = compare(ITEMS, BUDGET, MUST)
    assert [p["name"] for p in ranked][0] == "Acer Nitro V 15", "spec match + budget fit must win"
    assert ranked[-1]["name"] == "MacBook Pro M4", "over-budget must rank last"
    assert ranked[-1]["over_budget"] is True
    assert all(not p["over_budget"] for p in ranked[:-1])

    # a cheap machine that meets no requirement must lose to a pricier one that does
    assert score_product(ITEMS[0], BUDGET, MUST) > score_product(ITEMS[2], BUDGET, MUST)
    # no budget given: everything is comparable, nothing is flagged
    assert all(not p["over_budget"] for p in compare(ITEMS, 0, MUST))

    print("ok — 7 assertions passed")


if __name__ == "__main__":
    main()