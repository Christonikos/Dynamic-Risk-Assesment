#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
For the moment, this only contains toy examples to test the CI/CD part.
@author: christos
"""

# =============================================================================
# IMPORT MODULES
# =============================================================================


def add_numbers(x, y):
    return x + y


def test_add_numbers():
    assert add_numbers(2, 3) == 5
