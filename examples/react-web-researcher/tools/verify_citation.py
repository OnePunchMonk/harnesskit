"""Callback for the `verify_citation` tool. Stubbed — swap in a real fetch+check."""


def run(claim: str, source_url: str) -> str:
    return f"[stubbed: treated '{claim}' as verified against {source_url}]"
