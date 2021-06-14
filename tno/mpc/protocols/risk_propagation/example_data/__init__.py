"""
Module storing example data for the secure risk propagation protocol.
"""

import os

import pandas as pd

nodes_A = pd.read_csv(os.path.join(os.path.dirname(__file__), "nodes_A.csv"))
nodes_B = pd.read_csv(os.path.join(os.path.dirname(__file__), "nodes_B.csv"))
nodes_C = pd.read_csv(os.path.join(os.path.dirname(__file__), "nodes_C.csv"))

transactions_A = pd.read_csv(
    os.path.join(os.path.dirname(__file__), "transactions_A.csv")
)
transactions_B = pd.read_csv(
    os.path.join(os.path.dirname(__file__), "transactions_B.csv")
)
transactions_C = pd.read_csv(
    os.path.join(os.path.dirname(__file__), "transactions_C.csv")
)
transactions_all = pd.read_csv(
    os.path.join(os.path.dirname(__file__), "transactions_all.csv")
)
