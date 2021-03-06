"""
Configuration of a dictionary of bank accounts
"""
from __future__ import annotations

from typing import (
    Dict,
    Generator,
    Iterator,
    MutableMapping,
    Optional,
    Set,
    Union,
    ValuesView,
)

import pandas as pd

from tno.mpc.encryption_schemes.paillier import Paillier
from tno.mpc.protocols.distributed_keygen import DistributedPaillier

import tno.mpc.protocols.risk_propagation.bank  # to make sphinx find Bank correctly
from .account import Account


class Accounts(MutableMapping[str, Account]):
    """
    Class containing a dictionary of accounts
    """

    def __init__(self, accounts: Optional[Set[Account]] = None) -> None:
        """
        Initializes an Accounts instance

        :param accounts: optional set of accounts to initialise with
        """
        self._dict: Dict[str, Account] = (
            {account.label: account for account in accounts}
            if accounts is not None
            else dict()
        )

    def __delitem__(self, key: str) -> None:
        """
        Implements deletion of an account

        :param key: the key (label of the account) to delete
        """
        del self._dict[key]

    def __getitem__(self, key: str) -> Account:
        """
        Enables usage of [] to get an account

        :param key: the key (label of the account) to get
        :return: the found account
        """
        return self._dict[key]

    def __iter__(self) -> Iterator[str]:
        """
        :return: an iterator over the class
        """
        return iter(self._dict)

    def __len__(self) -> int:
        """
        :return: the number of accounts in the instance
        """
        return len(self._dict)

    def __setitem__(self, key: str, value: Account) -> None:
        """
        Enables usage of [] to set an account

        :param key: the key (label of the account) to set
        :param value: the account to set
        """
        self._dict[key] = value

    def __str__(self) -> str:
        """
        :return: string representation of the accounts, line-separated
        """
        return "\n".join(str(_) for _ in self.values())

    @classmethod
    def from_dataframe(
        cls,
        dataframe: pd.DataFrame,
        origin: tno.mpc.protocols.risk_propagation.bank.Bank,
        delta: float,
    ) -> Accounts:
        """
        Class method to create an instance of account from a Pandas dataframe

        :param dataframe: the dataframe to use in the initialisation with columns "id" and "score"
        :param origin: the originating bank
        :param delta: the delta to initialise the accounts with
        :return: an Accounts instance
        :raise ValueError: raised when provided dataframe does not contain the correct columns
        """
        if not {"id", "risk_score"}.issubset(dataframe.columns):
            raise ValueError(
                "Could not find the right columns in the provided dataframe"
            )
        accounts = set()
        for _, row in dataframe.iterrows():
            accounts.add(
                Account(
                    label=row["id"],
                    initial_risk_score=row["risk_score"],
                    origin=origin,
                    delta=delta,
                )
            )
        return cls(accounts)

    def encrypt(self, public_key: Union[DistributedPaillier, Paillier]) -> None:
        """
        Encrypt all accounts

        :param public_key: public key to use in the encryption
        """
        for account in self.values():
            account.encrypt(public_key)

    def get_accounts(
        self, account_keys: Optional[Set[str]] = None
    ) -> Union[ValuesView[Account], Generator[Account, None, None]]:
        """
        Retrieves all accounts or a subset if account_keys is provided

        :param account_keys: optional set of keys to retreive
        :return: generator of Accounts that are in account_keys
        """
        if account_keys is None:
            return self.values()
        return (self._dict[key] for key in account_keys if key in self.keys())
