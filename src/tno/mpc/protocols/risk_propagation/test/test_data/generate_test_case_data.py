"""
Script which can be used to generate new test data for the comparison between plaintext and secure risk propagation.
"""
from __future__ import annotations

import os
import random as random
from typing import Callable

import networkx as ntw  # type: ignore


def split_edges_over_nodes(
    number_of_nodes: int, banks: list[list[int]], edges: dict[tuple[int, int], int]
) -> list[list[tuple[int, str, int, str, int]]]:
    """
    Split the transaction edges over the banks. Each bank will receive a list of
    transactions, which are visible to them.

    :param number_of_nodes: The number of nodes that exists in the graph
    :param banks: A list of banks. Each bank consist of a list with node_id's in\
     their control
    :param edges: A dictionary from a tuple describing the directed edge to the\
     amount being transferred
    :return: A list for each bank. The list contains the tuple describing the\
     transactions `id_source, bank_source, id_destination, bank_destination, amount`
    """

    def get_bank_name(node_id: int) -> str:
        """
        Get the string of the bank to which the node belongs.

        :param node_id: the ID of the node
        :return: the name of the bank.
        """
        if 0 <= node_id < round(number_of_nodes / 3):
            return "local0"
        if node_id < round(number_of_nodes / 3) * 2:
            return "local1"
        return "local2"

    result = []
    for bank_nodes in banks:
        edges_in_view = [
            (
                edge[0],
                get_bank_name(edge[0]),
                edge[1],
                get_bank_name(edge[1]),
                edges[edge],
            )
            for edge in edges.keys()
            if edge[0] in bank_nodes or edge[1] in bank_nodes
        ]
        result.append(edges_in_view)

    return result


def write_transactions_per_bank(
    transactions_per_bank: list[list[tuple[int, str, int, str, int]]],
    file_name_generator: Callable[[int], str],
) -> None:
    """
    Write the transaction information to a csv for each bank. The format will be\
     'id_source,bank_source,id_destination,bank_destination,amount'.

    :param transactions_per_bank: For each bank a list with the tuple\
        corresponding to `id_source,bank_source,id_destination,bank_destination,amount`
    :param file_name_generator: Callable which is given the index of the bank\
         being processed and returns the file name as a string

    """
    for index, transaction_list in enumerate(transactions_per_bank):
        with open(file_name_generator(index), encoding="utf-8", mode="x") as file:
            file.write("id_source,bank_source,id_destination,bank_destination,amount\n")
            for transaction in transaction_list:
                file.write(
                    f"{transaction[0]},{transaction[1]},{transaction[2]},{transaction[3]},{transaction[4]}\n"
                )


def write_all_transactions(
    transactions: dict[tuple[int, int], int], file_name: str
) -> None:
    """
    Write all transactions to a single csv of the format\
        'id_source,id_destination,amount'.

    :param transactions: The transaction data in a dictonary of where the key is\
         a tuple of source and destination and the value is the amount.
    :param file_name: File name to which the information is store
    """
    with open(file_name, encoding="utf-8", mode="x") as file:
        file.write("id_source,id_destination,amount\n")
        for (id_source, id_destination), amount in transactions.items():
            file.write(f"{id_source},{id_destination},{amount}\n")


def write_bank_node_risk_scores(
    banks: list[list[int]],
    risk_scores: dict[int, float],
    file_name_generator: Callable[[int], str],
) -> None:
    """
    Write the node risk scores to a csv.

    :param banks: the list of bank names.
    :param risk_scores: a list of risk scores of the nodes of the banks.
    :param file_name_generator: Callable which is given the index of the bank\
         being processed and returns the file name as a string
    """
    for index, bank in enumerate(banks):
        with open(file_name_generator(index), encoding="utf-8", mode="x") as file:
            file.write("id,risk_score\n")
            for node in bank:
                file.write(f"{node},{risk_scores[node]}\n")


def generate_new_test_case(number_of_nodes: int, directory: str, periods: int) -> None:
    """
    Generate new test case data and store them in the directory. The test case\
         data has the corresponding format used in `load_file` (see `__init__.py`).

    :param number_of_nodes: The number of nodes in the transaction graph. Must\
         be a multiple of 3.
    :param directory: The directory to store the generated csv files.
    :param periods: The number of periods to generate.
    """
    assert number_of_nodes % 3 == 0, "Make the number of nodes a multiple of 3"

    bank_local0_nodes = [*range(round(number_of_nodes / 3))]
    bank_local1_nodes = [
        *range(round(number_of_nodes / 3), round(number_of_nodes / 3) * 2)
    ]
    bank_local2_nodes = [
        *range(round(number_of_nodes / 3) * 2, round(number_of_nodes / 3) * 3)
    ]
    banks_list = [bank_local0_nodes, bank_local1_nodes, bank_local2_nodes]
    nodes_file_name_generator = lambda bank_index: os.path.join(
        directory, f"nodes_local{bank_index}.csv"
    )

    node_risk_scores = {
        node: round(random.uniform(0, 1), 10) for node in range(number_of_nodes)
    }
    write_bank_node_risk_scores(banks_list, node_risk_scores, nodes_file_name_generator)
    write_bank_node_risk_scores(
        [[*range(number_of_nodes)]],
        node_risk_scores,
        lambda _: os.path.join(directory, "nodes_all.csv"),
    )

    for period in range(periods):
        graph = ntw.fast_gnp_random_graph(
            number_of_nodes, 3 / number_of_nodes, directed=True
        )
        directed_edges = list(graph.edges)
        random_amounts = [random.randint(100, 100000) for _ in directed_edges]
        edge_amounts_dict = dict(zip(directed_edges, random_amounts))
        transaction_file_name_generator = lambda bank_index: os.path.join(
            directory, f"transactions_local{bank_index}_period{period}.csv"
        )
        transactions_per_bank = split_edges_over_nodes(
            number_of_nodes, banks_list, edge_amounts_dict
        )
        write_transactions_per_bank(
            transactions_per_bank, transaction_file_name_generator
        )
        write_all_transactions(
            edge_amounts_dict,
            os.path.join(directory, f"transactions_all_period{period}.csv"),
        )
