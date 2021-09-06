"""
Configuration of a bank account
"""
from __future__ import annotations

from typing import Generator, Optional, Set, Tuple, Union, cast

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
        initial_risk_score: Optional[Union[float, PaillierCiphertext]] = None,
        delta: Optional[float] = None,
        transaction: Optional[Transaction] = None,
    ) -> None:
        """
        Initializes an instance of Account

        :param label: Identifier/label of the account
        :param origin: The bank that the account belongs to
        :param initial_risk_score: Optional initial risk_score of this account
        :param delta: Optional delta associated to the account
        :param transaction: Optionally specify a transaction already
        """
        self._label = label
        self._delta = delta
        self._origin = origin
        self._risk_score = initial_risk_score
        self._incoming_transactions: Transactions = Transactions()
        if transaction is not None:
            self.add_transaction(transaction)

    def __str__(self) -> str:
        """
        :return: String representation of the account information
        """
        return (
            f"Account information: {self.label} from {self.origin}."
            f" delta: {self._delta}, risk_score: {self._risk_score}"
        )

    @property
    def amounts(self) -> Generator[int, None, None]:
        """
        A generator of all transaction amounts

        :return: a generator of all transaction amounts
        """
        return (transaction.amount for transaction in self.transactions)

    @property
    def delta(self) -> float:
        """
        The delta that is used

        :return: delta
        :raise AttributeError: raised when delta is not set
        """
        if self._delta is None:
            raise AttributeError("Delta is undefined")
        return self._delta

    @property
    def label(self) -> str:
        """
        Label of the account

        :return: the label of the account
        """
        return self._label

    @property
    def linked_accounts(self) -> Set[Tuple[str, int]]:
        """
        Accounts linked to this account together with there amount value
        (label, amount)

        :return: a collection of all account linked to this account
        """
        return set(
            (transaction.sender, transaction.amount)
            for transaction in self.transactions
        )

    @property
    def origin(self) -> tno.mpc.protocols.risk_propagation.bank.Bank:
        """
        The bank that possesses the information on this account

        :return: a bank
        """
        return self._origin

    @property
    def risk_score(self) -> Union[float, PaillierCiphertext]:
        """
        Risk score of the account

        :return: current risk score
        :raise AttributeError: raised when risk score is not set
        """
        if self._risk_score is None:
            raise AttributeError("risk score is undefined")
        return self._risk_score

    @risk_score.setter
    def risk_score(self, risk_score: Union[float, PaillierCiphertext]) -> None:
        """
        Sets the risk score

        :param risk_score: a new risk score
        """
        self._risk_score = risk_score

    @property
    def safe_risk_score(self) -> PaillierCiphertext:
        """
        Encrypted risk score

        :return: current risk score
        :raise AttributeError: raised when risk score is not encrypted
        """
        if not isinstance(self._risk_score, PaillierCiphertext):
            raise AttributeError("risk score is not encrypted!")
        return cast(PaillierCiphertext, self.risk_score)

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
        return self._incoming_transactions

    def add_transaction(self, transaction: Transaction) -> None:
        """
        Adds incoming transaction to account

        :param transaction: the transaction to add
        """
        self._incoming_transactions.append(transaction)

    def encrypt(self, public_key: Union[DistributedPaillier, Paillier]) -> None:
        """
        Encrypt the risk score of the account

        :param public_key: the public key to use in the encryption
        :raise ValueError: raised when risk_score is already encrypted
        """
        if isinstance(self.risk_score, PaillierCiphertext):
            raise ValueError("Risk score is already encrypted")
        self.risk_score = public_key.encrypt(self.risk_score)
