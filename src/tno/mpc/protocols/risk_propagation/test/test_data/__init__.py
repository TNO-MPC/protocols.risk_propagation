"""
Module containing test data for usage in the tests.
"""
from __future__ import annotations

import csv
import os
from typing import TypedDict

import numpy as np
import numpy.typing as npt

from tno.mpc.protocols.risk_propagation.accounts import nodes_expected_dtype
from tno.mpc.protocols.risk_propagation.bank import transactions_expected_dtype


class InputData(TypedDict):
    """
    Structure defining all the neccessary information needed for a comparison
    between plaintext risk propagation and secure risk propagation.
    """

    numpy_nodes_A: npt.NDArray[np.object_]
    numpy_nodes_B: npt.NDArray[np.object_]
    numpy_nodes_C: npt.NDArray[np.object_]
    numpy_transactions_A: list[npt.NDArray[np.object_]]
    numpy_transactions_B: list[npt.NDArray[np.object_]]
    numpy_transactions_C: list[npt.NDArray[np.object_]]
    nodes_dict: dict[str, float]
    transactions_dict: list[dict[str, dict[str, int]]]
    transaction_totals_dict: list[dict[str, int]]


def load_nodes_numpy(file_path: str) -> npt.NDArray[np.object_]:
    """
    Load the csv file to a numpy array. The csv file must contain a header and
    have data in the form `id,risk_score`. Where `id` is a string and
    `risk_score` a float64.

    :param file_path: The file-path from which the node-data is loaded.
    :return: Numpy array representing the node information from the csv
    """
    return np.genfromtxt(
        file_path,
        dtype=nodes_expected_dtype,
        skip_header=1,
        delimiter=",",
    )


def load_transactions_of_period_numpy(file_path: str) -> npt.NDArray[np.object_]:
    """
    Load the csv file to a numpy array. The csv file must contain a header and
    have data in the form `id_source,bank_source,id_destination,bank_destination,amount`.
    Where `amount` is an int32 and all other fields are strings.

    :param file_path: The file-path from which the transaction-data is loaded.
    :return: Numpy array representing the transaction information from the csv
    """
    return np.genfromtxt(
        file_path,
        dtype=transactions_expected_dtype,
        skip_header=1,
        delimiter=",",
    )


def load_all_transactions_numpy(
    directory: str, bank_name: str, periods: int
) -> list[npt.NDArray[np.object_]]:
    """
    Load the transactions of period 0 till period `periods` for the specified
    bank. The files loaded must be in the directory and of the form
    `transactions_{bank_name}_period{period}.csv`.

    To learn more about the formatting of each file see
    `load_transactions_of_period_numpy`.

    :param directory: The directory to look for the files
    :param bank_name: The name of the bank
    :param periods: The number of periods to load
    :return: List of length period, where element `i` contains the transactions
         of period `i`.
    """
    result = []
    for period in range(periods):
        file_path = os.path.join(
            directory, f"transactions_{bank_name}_period{period}.csv"
        )
        result.append(load_transactions_of_period_numpy(file_path))
    return result


def read_nodes_risk_score_dict(file_path: str) -> dict[str, float]:
    """
    Read node id and risk score from the file to a dictionary.
    The file must contain a header and be of the form `id,risk_score`.
    Where `id` is a string and `risk_score` a float.

    :param file_path: File to read the information from.
    :return: Dictionary containing the node id's and their initial risk score.
    """
    with open(file_path, encoding="utf-8") as csv_file:
        csv_reader = csv.reader(csv_file)
        # skip the header line
        next(csv_reader)
        return {row[0]: float(row[1]) for row in csv_reader}


def load_transactions_dict(file_path: str) -> dict[str, dict[str, int]]:
    """
    Read the transactions of the file to a dictionary. The file must contain a
    header. A transaction consists of
    `id_source,id_destination,amount`. Where the amount is an integer and the ids
    are strings.

    A transaction becomes the following entry in the dictionary
    `{id_destination: {id_source:amount}}`.

    :param file_path: The file to read the transactions from.
    :return: Dictionary mapping from `id_destination` to a dictionary containing the `id_source` with the amount
    """
    with open(file_path, encoding="utf-8") as csv_file:
        csv_reader = csv.reader(csv_file)
        # skip the header line
        next(csv_reader)
        result: dict[str, dict[str, int]] = {}
        for row in csv_reader:
            if row[1] not in result:
                result[row[1]] = {row[0]: int(row[2])}
            else:
                result[row[1]][row[0]] = int(row[2])
        return result


def load_all_transactions_dict(
    directory: str, periods: int
) -> list[dict[str, dict[str, int]]]:
    """
    Load the transactions of period 0 till period `periods` bank. The files
    loaded must be in the directory and of the form `transactions_all_period{period}.csv`.

    To learn more about the formatting of each file see
    `load_transactions_dict`.

    :param directory: The directory to look for the files
    :param periods: The number of periods to load
    :return: List of length period, where element `i` contains the transactions
         of period `i`.
    """
    result = []
    for period in range(periods):
        file_path = os.path.join(directory, f"transactions_all_period{period}.csv")
        result.append(load_transactions_dict(file_path))
    return result


def load_files(
    directory: str, bank_names: tuple[str, str, str], periods: int
) -> InputData:
    """
    Load all the transactions and node information in formats used by
    the plaintext risk propagation and the secure risk propagation.

    :param directory: The directory in which the files are stored.
    :param bank_names: The names of the banks. The bank name must correspond
        to the name in the pool and the bank name in the test case files.
    :param periods: The number of periods to load.

    :return: InputData corresponding to the inputs needed for plaintext and
        secure risk propagation.
    """
    nodes_a = load_nodes_numpy(os.path.join(directory, f"nodes_{bank_names[0]}.csv"))
    nodes_b = load_nodes_numpy(os.path.join(directory, f"nodes_{bank_names[1]}.csv"))
    nodes_c = load_nodes_numpy(os.path.join(directory, f"nodes_{bank_names[2]}.csv"))

    transactions_a = load_all_transactions_numpy(directory, bank_names[0], periods)
    transactions_b = load_all_transactions_numpy(directory, bank_names[1], periods)
    transactions_c = load_all_transactions_numpy(directory, bank_names[2], periods)

    transactions_all = load_all_transactions_dict(directory, periods)

    transactions_totals_all = []

    for transaction_period in transactions_all:
        transactions_totals_all.append(
            {k: sum(v.values()) for k, v in transaction_period.items()}
        )

    return {
        "numpy_nodes_A": nodes_a,
        "numpy_nodes_B": nodes_b,
        "numpy_nodes_C": nodes_c,
        "numpy_transactions_A": transactions_a,
        "numpy_transactions_B": transactions_b,
        "numpy_transactions_C": transactions_c,
        "nodes_dict": read_nodes_risk_score_dict(
            os.path.join(directory, "nodes_all.csv")
        ),
        "transactions_dict": transactions_all,
        "transaction_totals_dict": transactions_totals_all,
    }
