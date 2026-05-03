"""Tiny console helpers used by demos."""

from __future__ import annotations


def banner(title: str) -> None:
    line = "=" * len(title)
    print(f"\n{line}\n{title}\n{line}")


def kv(key: str, value: object) -> None:
    print(f"  {key:<24}: {value}")


def shorten(text: str, width: int = 90) -> str:
    text = " ".join(str(text).split())
    if len(text) <= width:
        return text
    return text[: width - 3] + "..."

