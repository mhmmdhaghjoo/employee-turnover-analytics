"""A small synthetic file in the same format as the real dataset, so tests and CI
run without the data (which is not redistributed in this repository)."""

from __future__ import annotations

from pathlib import Path

import pytest

from turnover.demo import make_raw


@pytest.fixture(scope="session")
def raw_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("data") / "hr.csv"
    make_raw().to_csv(path, index=False)
    return path
