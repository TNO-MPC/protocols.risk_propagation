"""
Module storing example data for the secure risk propagation protocol.
"""

import os

from tno.mpc.protocols.risk_propagation.test.test_data import load_files

input_data = load_files(os.path.dirname(__file__), ("local0", "local1", "local2"), 1)
