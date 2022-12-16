"""
Configuration of a bank
"""
import logging
from typing import Dict, Optional, Set, Union

import numpy as np
import numpy.typing as npt

from tno.mpc.encryption_schemes.paillier import Paillier, PaillierCiphertext
from tno.mpc.protocols.distributed_keygen import DistributedPaillier

from .account import Account
from .accounts import Accounts
from .transaction import Transaction
from .transactions import Transactions

logger = logging.getLogger(__name__)


class Bank:
    """
    Class containing the configuration of a single bank
    """

    def __init__(
        self, name: str, transactions: Optional[npt.NDArray[np.object_]] = None
    ) -> None:
        """
        Initializes a bank instance

        :param name: name of the bank
        :param transactions: optional dataframe with transactions to process
        """
        self._name = name
        self._accounts = Accounts()
        self._imported_accounts = Accounts()
        self._external_transactions = Transactions()
        if transactions is not None:
            self.process_transactions(transactions)

    def __str__(self) -> str:
        """
        :return: string representation of the bank
        """
        return f"Bank: {self.name}"

    @property
    def accounts(self) -> Set[str]:
        """
        The collection of account labels of this bank

        :return: set of account labels of this bank
        """
        return set(self._accounts.keys())

    @accounts.setter
    def accounts(self, accounts: Accounts) -> None:
        """
        Sets the accounts of this bank

        :param accounts: an Accounts instance
        """
        self._accounts = accounts

    @property
    def accounts_dict(self) -> Accounts:
        """
        The collection of accounts of this bank

        :return: the accounts of this bank
        """
        return self._accounts

    @property
    def external_accounts(self) -> Set[str]:
        """
        The collection of external accounts

        :return: set of external accounts of this bank
        """
        return set(_.sender for _ in self._external_transactions)

    @property
    def name(self) -> str:
        """
        The name of the bank

        :return: the name of the bank instance
        """
        return self._name

    @property
    def risk_scores(self) -> Dict[str, PaillierCiphertext]:
        """
        A dictionary of risk scores per account of this bank

        :return: a dictionary of all risk scores of this bank
        """
        scores = self.get_risk_scores()
        imported_scores = self.get_imported_risk_scores()
        return {**scores, **imported_scores}

    def encrypt(self, public_key: Union[DistributedPaillier, Paillier]) -> None:
        """
        Encrypts risk scores of all accounts of this bank

        :param public_key: the public key used in the encryption
        """
        self._accounts.encrypt(public_key)

    def get_imported_risk_scores(
        self, account_keys: Optional[Set[str]] = None
    ) -> Dict[str, PaillierCiphertext]:
        """
        Gets the imported accounts risk scores

        :param account_keys: the scores to retrieve
        :return: dict of risk scores
        """
        accounts = self._imported_accounts.get_accounts(account_keys)
        return {account.label: account.safe_risk_score for account in accounts}

    def get_risk_scores(
        self, account_keys: Optional[Set[str]] = None
    ) -> Dict[str, PaillierCiphertext]:
        """
        Gets the accounts risk scores

        :param account_keys: the scores to retrieve
        :return: dict of risk scores
        """
        accounts = self._accounts.get_accounts(account_keys)
        return {account.label: account.safe_risk_score for account in accounts}

    def process_accounts(
        self, array: npt.NDArray[np.object_], delta: float = 0.5
    ) -> None:
        """
        Initialises the accounts belonging to this bank

        :param array: an array containing the accounts with risk scores
        :param delta: the delta to be used
        """
        self._accounts = Accounts.from_numpy_array(array, delta=delta, origin=self)

    def process_transactions(self, array: npt.NDArray[np.object_]) -> None:
        """
        Processes the transaction array

        :param array: array with the following columns and types;
            "id_source" (unicode at most 100 char),
            "id_destination" (unicode at most 100 char),
            "bank_source" (unicode at most 100 char),
            "bank_destination (unicode at most 100 char)",
            "amount" (int32)
        :raise ValueError: raised when provided array does not contain the correct columns
        """

        expected_dtype = np.dtype(
            [
                ("id_source", np.unicode_, 100),
                ("bank_source", np.unicode_, 100),
                ("id_destination", np.unicode_, 100),
                ("bank_destination", np.unicode_, 100),
                ("amount", np.int32),
            ]
        )
        typed_array = np.rec.array(array, dtype=expected_dtype)

        def __handle_single_transaction(
            sender: str,
            receiver: str,
            amount: int,
            origin_bank: str,
            destination_bank: str,
            accounts: Set[str],
        ) -> None:
            """
            Process a single transaction

            :param sender: id of the sender account
            :param receiver: id of the receiver account
            :param amount: amount of the transaction
            :param origin_bank: id of the bank from the sender account
            :param destination_bank: id of the bank from the receiver account
            :param accounts: Set of accounts owned by this bank
            """
            transaction = Transaction(sender, receiver, int(amount))
            if destination_bank == self.name:
                if receiver not in accounts:
                    self._accounts[receiver] = Account(
                        receiver, origin=self, transaction=transaction
                    )
                    # keep track of the accounts of this bank
                    accounts.add(receiver)
                else:
                    self._accounts[receiver].add_transaction(transaction)
                if origin_bank != self.name:
                    self._external_transactions.append(transaction)
                    self._imported_accounts[sender] = Account(
                        sender, origin=self, transaction=transaction
                    )

        # self.account can be an expensive operation (turning list to a set)
        # therefore the current_accounts set is created once and kept up to date with new accounts
        current_accounts = self.accounts
        logger.info(f"Processing {typed_array.size} transactions")
        for transaction in typed_array:
            __handle_single_transaction(
                transaction.id_source.item(),
                transaction.id_destination.item(),
                transaction.amount.item(),
                transaction.bank_source.item(),
                transaction.bank_destination.item(),
                accounts=current_accounts,
            )

    def set_risk_score(
        self, account: str, risk_score: PaillierCiphertext, external: bool = False
    ) -> None:
        """
        Sets the risk score of an account in this bank

        :param account: the name of the account
        :param risk_score: the new risk score
        :param external: optional boolean, set to true for imported scores
        """
        if external:
            self._imported_accounts[account].risk_score = risk_score
        else:
            self._accounts[account].risk_score = risk_score
