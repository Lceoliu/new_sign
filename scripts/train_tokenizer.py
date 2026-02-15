#!/usr/bin/env python3
"""Wrapper script to train pose tokenizer."""

import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from pose_tokenizer.scripts.train import main


if __name__ == "__main__":
    main()
