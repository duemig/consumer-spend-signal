"""Fail if a number quoted in the README no longer matches results/summary.json.

The study writes summary.json; the README quotes from it. This test reformats the
headline figures and every variant's mean IC and net Sharpe the way the README
prints them and checks each string is present, so the two cannot drift apart.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _present(readme: str, value: str) -> bool:
    """Whether ``value`` appears in ``readme`` not glued to another digit.

    The boundaries reject an adjacent digit, so ``0.77`` does not match inside
    ``0.775``, but allow ordinary punctuation such as a sentence-ending period.
    """
    return re.search(r"(?<!\d)" + re.escape(value) + r"(?!\d)", readme) is not None


def _expected_numbers(summary: dict) -> list[str]:
    numbers = [
        f"{summary['oracle_ic']['mean']:.4f}",
        f"{summary['final_ic']['mean']:.4f}",
        f"{summary['final_net_sharpe']['mean']:.2f}",
        f"{summary['final_net_sharpe']['lo']:.2f}",
        f"{summary['final_net_sharpe']['hi']:.2f}",
        f"{summary['final_recovery']['mean']:.0%}",
        f"{summary['final_recovery']['min']:.0%}",
        f"{summary['final_recovery']['max']:.0%}",
        f"{summary['final_sharpe_min']:.2f}",
        f"{summary['final_sharpe_max']:.2f}",
        f"{summary['final_t_min']:.1f}",
        f"{summary['final_t_max']:.1f}",
    ]
    for row in summary["distribution"].values():
        numbers.append(f"{row['ic_mean']:.4f}")
        numbers.append(f"{row['sharpe_mean']:.2f}")
    return numbers


def test_readme_numbers_match_summary():
    readme = (ROOT / "README.md").read_text()
    summary = json.loads((ROOT / "results" / "summary.json").read_text())
    missing = sorted({value for value in _expected_numbers(summary) if not _present(readme, value)})
    assert not missing, f"README.md is out of date with results/summary.json; missing values: {missing}"
