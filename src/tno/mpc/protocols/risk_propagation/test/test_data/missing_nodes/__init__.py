"""
Module storing example data for the secure risk propagation protocol.
For this test case two transactions are inconsistent.
- The transaction from node 1 to 8 in period 0 is expected by party `local2`, but the objective indicator of node 1 is not sent by party `local0`.
- The transaction from node 2 to 5 in period 1 is not expected by party `local1`, but the objective indicator of node 2 is sent by party `local0`.
"""

import os

from tno.mpc.protocols.risk_propagation.test.test_data import load_files

input_data = load_files(os.path.dirname(__file__), ("local0", "local1", "local2"), 1)
