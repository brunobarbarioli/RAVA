from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class VerificationConfig:
    name: str
    pre: bool
    runtime: bool
    posthoc: bool


BASELINES = {
    "none": VerificationConfig(name="none", pre=False, runtime=False, posthoc=False),
    "pre": VerificationConfig(name="pre", pre=True, runtime=False, posthoc=False),
    "runtime": VerificationConfig(name="runtime", pre=False, runtime=True, posthoc=False),
    "posthoc": VerificationConfig(name="posthoc", pre=False, runtime=False, posthoc=True),
    "full": VerificationConfig(name="full", pre=True, runtime=True, posthoc=True),
    "guardrails_placeholder": VerificationConfig(
        name="guardrails_placeholder", pre=True, runtime=True, posthoc=False
    ),
}


def get_verification_config(name: str) -> VerificationConfig:
    key = name.lower()
    if key not in BASELINES:
        raise ValueError(f"Unknown verification config: {name}")
    return BASELINES[key]
