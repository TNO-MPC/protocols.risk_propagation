"""
Plaintext version of the risk propagation algorithm.

Preprocessing should be done in the following way:

1.  Select some period "Y" over which we are going to propagate the risk scores and take all the nodes that made at least one transaction in this period. For example, 4 weeks.
2.  For all these nodes, select some period "X" over which we compute the initial risk scores. For example, the same 4 weeks as Y as well as the 4 weeks before Y. This should result in a dictionary that maps every node to an initial risk score.
3.  Define a period "Z" with which we are going to "slice" the period Y for a more fine-grained analysis. This can for example be 4 times 1 week, to lead to 4 "Z's".
4.  For each period Z, compute a (directed) transaction graph for that period by connecting nodes that send money to other nodes. As edge labels we use the amount of money transferred. If multiple transactions take place from one node to the next node, aggregate them in one edge and sum the amounts.
    Ultimately, this should be represented as a dictionary that maps each node in period Z to a dictionary with as keys the other nodes that made a transaction to this node and as value the (aggregated) amount transferred from that node. Put the dictionaries for all Z's (for example 4) into a list in the order of the Z's "dics_transactions".
5.  For each period Z, compute for each node in the transaction network from step 4 the total amount of money received by this node and put it in a dictionary mapping the node to the total amount received. Put these in a list "dics_total" that matches the order of the Z's and the order of the list resulting from step 4.
"""
from __future__ import annotations

import copy
from typing import Callable

import pandas as pd


def compute_delta(iteration: int, a_val: float = 0.5, b_val: float = 0.5) -> float:
    """
    Compute the delta given an a, b and the current iteration.

    :param iteration: The iteration for which the delta is needed.
    :param a_val: The a value for the delta function.
    :param b_val: The b value for the delta function.
    :return: The result of `a/(iteration^b)`
    """
    return float(a_val / (iteration**b_val))


def risk_propagation(
    dic_risk_scores: dict[str, float],
    dics_transactions: list[dict[str, dict[str, int]]],
    dics_total: list[dict[str, int]],
    delta_func: Callable[[int], float],
    nr_iter: int = 2,
) -> tuple[dict[str, float], pd.DataFrame]:
    """
    The main risk propagation function. Given the three dictionaries output by the preprocessing, perform a number of
    iterations of the risk propagation algorithms. Store intermediate results in a pandas dataframe (at each node,
    the computed risk scores of all the identities).

    :param dic_risk_scores: A dictionary that maps identities to risk scores.
    :param dics_transactions: A list of dictionaries that map, for each identity (node) in the transaction network, to a dictionary containing as keys the nodes that made a transaction to this node and as values the amount transferred. Each dictionary should correspond to one period Z (the list is assumed to be ordered, so $z_0$ should be also be in the first spot in the list).
    :param dics_total: A list of dictionaries that map for each identity to the total amount of money received in a period Z. The indices in the list should correspond to the same indices in the transaction dictionaries.
    :param delta_func: The delta function to use.
    :param nr_iter: The number of iterations to perform.
    :return: A tuple where the first element represents the resulting risk scores and the second element the dataframe containing all the intermediate risk scores.
    """
    # run risk propagation for the number of iterations
    risk_scores = copy.deepcopy(dic_risk_scores)

    # Create results dataframe
    results = pd.DataFrame(list(risk_scores.items()), columns=["node_id", "risk_0"])

    for index, transactions in enumerate(dics_transactions):
        totals = dics_total[index]
        for iteration in range(1, nr_iter + 1):
            # run an iteration with the new delta
            delta = delta_func(iteration)
            risk_scores = do_iteration(risk_scores, transactions, totals, delta)

            # Store results
            results[f"risk_z{index}_{iteration}"] = results["node_id"].map(risk_scores)

    # Convert to floats for faster post-processing
    results["risk_0"] = results["risk_0"].astype(float)

    for i in range(len(dics_transactions)):
        for j in range(1, nr_iter + 1):
            results[f"risk_z{i}_{j}"] = results[f"risk_z{i}_{j}"].astype(float)

    return risk_scores, results


def do_iteration(
    dic_risk_scores: dict[str, float],
    dic_transactions: dict[str, dict[str, int]],
    dic_total: dict[str, int],
    delta: float,
) -> dict[str, float]:
    """
    Perform one iteration of the risk propagation algorithm. Computes, for every node, the new risk scores given the transaction network and the risk scores of the neighbors.

    :param dic_risk_scores: A dictionary that maps identities to risk scores.
    :param dic_transactions: A dictionary that maps, for each identity (node) in the transaction network, to a dictionary containing as keys the nodes that made a transaction to this node and as values the amount transferred.
    :param dic_total: A dictionary that maps for each identity to the total amount of money received in period Y.
    :param delta: The delta value to use in this iteration.
    :return: A dictionary with the same structure as dic_risk_scores containing the new risk score values.
    """
    dic_risk_scores_updated = {}
    for node in dic_risk_scores:
        old_risk = dic_risk_scores[node]
        total = dic_total[node] if node in dic_total else 0
        if total == 0:
            weighted_risk = 0.0
        else:
            inc_trans = dic_transactions[node]
            weighted_risk = 0.0
            for source in inc_trans:
                risk_source = dic_risk_scores.get(source, 0.0)
                amount = inc_trans[source]
                weighted_risk += risk_source * amount
            weighted_risk = weighted_risk / total
        new_risk = (1 - delta) * old_risk + delta * weighted_risk
        dic_risk_scores_updated[node] = new_risk
    return dic_risk_scores_updated
