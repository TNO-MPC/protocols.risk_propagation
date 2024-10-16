"""
Configuration of a bank
"""

from __future__ import annotations

import logging

import numpy as np
import numpy.typing as npt

from tno.mpc.encryption_schemes.paillier import Paillier, PaillierCiphertext
from tno.mpc.protocols.distributed_keygen import DistributedPaillier

from .account import Account
from .accounts import Accounts
from .transaction import Transaction
from .transactions import Transactions

logger = logging.getLogger(__name__)

# the expected data type for the transactions
transactions_expected_dtype = np.dtype(
    [
        ("id_source", np.str_, 100),
        ("bank_source", np.str_, 100),
        ("id_destination", np.str_, 100),
        ("bank_destination", np.str_, 100),
        ("amount", np.int32),
    ]
)


class Bank:
    """
    Class containing the configuration of a single bank
    """

    def __init__(
        self,
        name: str,
        transactions: list[npt.NDArray[np.object_]] | None = None,
        n_periods: int = 1,
    ) -> None:
        """
        Initializes a bank instance

        :param name: name of the bank
        :param transactions: optional dataframe with transactions to process
        """
        self._name = name
        self._accounts = Accounts()
        self._imported_accounts = Accounts()
        self._n_periods = n_periods
        self._all_external_transactions = [Transactions() for _ in range(n_periods)]
        self._current_external_transactions = self._all_external_transactions[0]
        if transactions is not None:
            self.process_transactions(transactions)

    def __str__(self) -> str:
        """
        :return: string representation of the bank
        """
        return f"Bank: {self.name}"

    @property
    def accounts(self) -> set[str]:
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
    def external_accounts(self) -> set[str]:
        """
        The collection of external accounts, i.e. accounts of this bank that
        are involved in a transaction with an account from another bank.

        This set changes when the period changes.

        :return: set of external accounts of this bank
        """
        return {_.sender for _ in self._current_external_transactions}

    @property
    def name(self) -> str:
        """
        The name of the bank

        :return: the name of the bank instance
        """
        return self._name

    @property
    def risk_scores(self) -> dict[str, PaillierCiphertext]:
        """
        A dictionary of risk scores per account of this bank. If the account has a risk score.

        :return: a dictionary of all risk scores of this bank
        """
        scores = self.get_risk_scores()
        imported_scores = self.get_imported_risk_scores()
        return {**scores, **imported_scores}

    def encrypt(self, public_key: DistributedPaillier | Paillier) -> None:
        """
        Encrypts risk scores of all accounts of this bank

        :param public_key: the public key used in the encryption
        """
        self._accounts.encrypt(public_key)

    def get_imported_risk_scores(
        self, account_keys: set[str] | None = None
    ) -> dict[str, PaillierCiphertext]:
        """
        Gets the encrypted risk scores of the imported accounts. If the account has no encrypted risk score no entry is added.

        :param account_keys: the scores to retrieve
        :return: dict of encrypted risk scores
        """
        accounts = self._imported_accounts.get_accounts(account_keys)
        return {
            account.label: account.unsafe_encrypted_risk_score
            for account in accounts
            if account.has_encrypted_risk_score
        }

    def get_risk_scores(
        self, account_keys: set[str] | None = None
    ) -> dict[str, PaillierCiphertext]:
        """
        Gets the accounts risk scores

        :param account_keys: the scores to retrieve
        :return: dict of risk scores
        """
        accounts = self._accounts.get_accounts(account_keys)
        return {
            account.label: account.unsafe_encrypted_risk_score for account in accounts
        }

    def set_current_period(self, period_z: int) -> None:
        """
        Set the period z that should be used for the next iteration(s).

        :param period_z: the period that should be used.
        """
        self._current_external_transactions = self._all_external_transactions[period_z]
        for account in self._accounts.values():
            account.current_incoming_transactions = account.all_incoming_transactions[
                period_z
            ]

    def process_accounts(self, array: npt.NDArray[np.object_]) -> None:
        """
        Initialises the accounts belonging to this bank

        :param array: an array containing the accounts with risk scores
        """
        self._accounts = Accounts.from_numpy_array(
            array, origin=self, periods=self._n_periods
        )

    def process_transactions(self, array: list[npt.NDArray[np.object_]]) -> None:
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
        assert len(array) == len(self._all_external_transactions)

        typed_array = [
            np.rec.array(array[i], dtype=transactions_expected_dtype)
            for i in range(len(array))
        ]

        def __handle_single_transaction(
            sender: str,
            receiver: str,
            amount: int,
            origin_bank: str,
            destination_bank: str,
            accounts: set[str],
            period: int,
        ) -> None:
            """
            Process a single transaction

            :param sender: id of the sender account
            :param receiver: id of the receiver account
            :param amount: amount of the transaction
            :param origin_bank: id of the bank from the sender account
            :param destination_bank: id of the bank from the receiver account
            :param accounts: Set of accounts owned by this bank
            :param period: The period in which this transaction was made
            """
            transaction = Transaction(sender, receiver, int(amount))
            if destination_bank == self.name:
                if receiver not in accounts:
                    self._accounts[receiver] = Account(
                        receiver,
                        origin=self,
                        transaction=transaction,
                        period=period,
                        n_periods=self._n_periods,
                    )
                    # keep track of the accounts of this bank
                    accounts.add(receiver)
                else:
                    self._accounts[receiver].add_transaction(transaction, period=period)
                if origin_bank != self.name:
                    self._all_external_transactions[period].append(transaction)
                    self._imported_accounts[sender] = Account(
                        sender,
                        origin=self,
                        transaction=transaction,
                        period=period,
                        n_periods=self._n_periods,
                    )

        # self.account can be an expensive operation (turning list to a set)
        # therefore the current_accounts set is created once and kept up to date with new accounts
        current_accounts = self.accounts
        logger.info(
            f"Processing {sum(len(period) for period in typed_array)} transactions"
        )
        for period, transactions in enumerate(typed_array):
            for transaction in transactions:
                __handle_single_transaction(
                    transaction.id_source.item(),
                    transaction.id_destination.item(),
                    transaction.amount.item(),
                    transaction.bank_source.item(),
                    transaction.bank_destination.item(),
                    accounts=current_accounts,
                    period=period,
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
