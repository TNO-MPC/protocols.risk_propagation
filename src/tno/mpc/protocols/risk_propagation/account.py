"""
Configuration of a bank account
"""
from __future__ import annotations

from typing import Generator, cast

from tno.mpc.encryption_schemes.paillier import Paillier, PaillierCiphertext
from tno.mpc.protocols.distributed_keygen import DistributedPaillier

import tno.mpc.protocols.risk_propagation.bank  # to make sphinx find Bank correctly
from .transaction import Transaction
from .transactions import Transactions


class Account:
    """
    Class containing information on a bank account
    """

    def __init__(
        self,
        label: str,
        origin: tno.mpc.protocols.risk_propagation.bank.Bank,
        initial_risk_score: float | PaillierCiphertext | None = None,
        transaction: Transaction | None = None,
        period: int | None = None,
        n_periods: int = 1,
    ) -> None:
        """
        Initializes an instance of Account

        :param label: Identifier/label of the account
        :param origin: The bank that the account belongs to
        :param initial_risk_score: Optional initial risk_score of this account
        :param transaction: Optionally specify a transaction already
        :param period: The period z to which the transaction belongs.
        :param n_periods: The total number of periods z in the protocol.
        """
        self._label = label
        self._origin = origin
        self._risk_score = initial_risk_score
        self.all_incoming_transactions = [Transactions() for _ in range(n_periods)]
        self.current_incoming_transactions = self.all_incoming_transactions[0]
        if transaction is not None and period is not None:
            self.add_transaction(transaction, period)

    def __str__(self) -> str:
        """
        :return: String representation of the account information
        """
        return (
            f"Account information: {self.label} from {self.origin}."
            f"risk_score: {self._risk_score}"
        )

    @property
    def amounts(self) -> Generator[int, None, None]:
        """
        A generator of all transaction amounts

        :return: a generator of all transaction amounts
        """
        return (transaction.amount for transaction in self.transactions)

    @property
    def label(self) -> str:
        """
        Label of the account

        :return: the label of the account
        """
        return self._label

    @property
    def linked_accounts(self) -> set[tuple[str, int]]:
        """
        Accounts linked to this account together with their amount value
        (label, amount)

        :return: a collection of all account linked to this account
        """
        return {
            (transaction.sender, transaction.amount)
            for transaction in self.transactions
        }

    @property
    def origin(self) -> tno.mpc.protocols.risk_propagation.bank.Bank:
        """
        The bank that possesses the information on this account

        :return: a bank
        """
        return self._origin

    @property
    def risk_score(self) -> float | PaillierCiphertext:
        """
        Risk score of the account

        :return: current risk score
        :raise AttributeError: raised when risk score is not set
        """
        if self._risk_score is None:
            raise AttributeError("risk score is undefined")
        return self._risk_score

    @risk_score.setter
    def risk_score(self, risk_score: float | PaillierCiphertext) -> None:
        """
        Sets the risk score

        :param risk_score: a new risk score
        """
        self._risk_score = risk_score

    @property
    def unsafe_encrypted_risk_score(self) -> PaillierCiphertext:
        """
        Encrypted risk score. Note that the ciphertext is not (yet) safe to
        broadcast as it first needs to be randomized.

        :return: current risk score
        :raise AttributeError: raised when risk score is not encrypted
        """
        if not isinstance(self._risk_score, PaillierCiphertext):
            raise AttributeError("risk score is not encrypted!")
        return cast(PaillierCiphertext, self.risk_score)

    @property
    def has_encrypted_risk_score(self) -> bool:
        """
        Check if this account's risk score is a paillier ciphertext

        :return: True if the risk score is encrypted else false
        """
        return isinstance(self._risk_score, PaillierCiphertext)

    @property
    def total_income(self) -> int:
        """
        Sum of all incoming transactions

        :return: total amount of incoming transactions
        """
        return sum(self.amounts)

    @property
    def transactions(self) -> Transactions:
        """
        The collection of all incoming transactions

        :return: the collection of incoming transactions
        """
        return self.current_incoming_transactions

    def add_transaction(self, transaction: Transaction, period: int) -> None:
        """
        Adds incoming transaction to account

        :param transaction: the transaction to add
        :param period: the period z to which the transaction belongs.
        """
        self.all_incoming_transactions[period].append(transaction)

    def encrypt(self, public_key: DistributedPaillier | Paillier) -> None:
        """
        Encrypt the risk score of the account

        :param public_key: the public key to use in the encryption
        :raise ValueError: raised when risk_score is already encrypted
        """
        if isinstance(self.risk_score, PaillierCiphertext):
            raise ValueError("Risk score is already encrypted")
        self.risk_score = public_key.unsafe_encrypt(self.risk_score)
