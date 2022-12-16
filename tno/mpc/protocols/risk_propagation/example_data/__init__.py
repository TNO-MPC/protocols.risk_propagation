"""
Module storing example data for the secure risk propagation protocol.
"""

import os

import numpy as np
import numpy.typing as npt

nodes_datatype = np.dtype([("id", np.unicode_, 100), ("risk_score", np.float64)])
transaction_datatype = np.dtype(
    [
        ("id_source", np.unicode_, 100),
        ("bank_source", np.unicode_, 100),
        ("id_destination", np.unicode_, 100),
        ("bank_destination", np.unicode_, 100),
        ("amount", np.int32),
    ]
)
nodes_A: npt.NDArray[np.object_] = np.genfromtxt(
    os.path.join(os.path.dirname(__file__), "nodes_A.csv"),
    dtype=nodes_datatype,
    skip_header=1,
    delimiter=",",
)
nodes_B: npt.NDArray[np.object_] = np.genfromtxt(
    os.path.join(os.path.dirname(__file__), "nodes_B.csv"),
    dtype=nodes_datatype,
    skip_header=1,
    delimiter=",",
)
nodes_C: npt.NDArray[np.object_] = np.genfromtxt(
    os.path.join(os.path.dirname(__file__), "nodes_C.csv"),
    dtype=nodes_datatype,
    skip_header=1,
    delimiter=",",
)

transactions_A: npt.NDArray[np.object_] = np.genfromtxt(
    os.path.join(os.path.dirname(__file__), "transactions_A.csv"),
    dtype=transaction_datatype,
    skip_header=1,
    delimiter=",",
)
transactions_B: npt.NDArray[np.object_] = np.genfromtxt(
    os.path.join(os.path.dirname(__file__), "transactions_B.csv"),
    dtype=transaction_datatype,
    skip_header=1,
    delimiter=",",
)
transactions_C: npt.NDArray[np.object_] = np.genfromtxt(
    os.path.join(os.path.dirname(__file__), "transactions_C.csv"),
    dtype=transaction_datatype,
    skip_header=1,
    delimiter=",",
)
transactions_all: npt.NDArray[np.object_] = np.genfromtxt(
    os.path.join(os.path.dirname(__file__), "transactions_all.csv"),
    dtype=transaction_datatype,
    skip_header=1,
    delimiter=",",
)
