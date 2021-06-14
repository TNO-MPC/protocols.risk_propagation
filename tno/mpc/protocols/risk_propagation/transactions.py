"""
Configuration of a list of transactions
"""
from __future__ import annotations

from typing import Iterator, List, Optional

from .transaction import Transaction


class Transactions:
    """
    Class containing a list of transactions
    """

    def __init__(self, transactions: Optional[List[Transaction]] = None) -> None:
        """
        Initializes a transaction list

        :param transactions: Optional list of transactions
        """
        self._list: List[Transaction] = transactions if transactions is not None else []

    def __iter__(self) -> Iterator[Transaction]:
        """
        :return: iterator over the transaction list
        """
        return iter(self._list)

    def __str__(self) -> str:
        """
        :return: string representation of transactions, line-separated
        """
        return "\n".join(str(_) for _ in self._list)

    def append(self, transaction: Transaction) -> None:
        """
        Adds a transaction to the list

        :param transaction: a transaction to add to the list
        """
        self._list.append(transaction)
