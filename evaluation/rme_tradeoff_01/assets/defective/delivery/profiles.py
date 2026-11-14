"""Operational profiles (protected). Budgets apply to each deliver() operation."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Profile:
    name: str
    reconcile_budget: int  # authoritative lookups per operation
    delivery_budget: int   # sends per operation


PROFILES = {
    "live": Profile("live", reconcile_budget=2, delivery_budget=4),
    "safe": Profile("safe", reconcile_budget=1, delivery_budget=2),
}


def get_profile(name):
    if name not in PROFILES:
        raise ValueError(f"unknown profile {name!r}")
    return PROFILES[name]
