"""Backward-compatible wrapper for the renamed dataset transform CLI."""

from __future__ import annotations

import sys

import dataset_transform


if __name__ == "__main__":
    dataset_transform.main(["repair-instruction", *sys.argv[1:]])
