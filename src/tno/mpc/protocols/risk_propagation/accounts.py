"""
Configuration of a dictionary of bank accounts
"""

from __future__ import annotations

import logging
from collections.abc import Generator, Iterator, MutableMapping, ValuesView

import numpy as np
import numpy.typing as npt

from tno.mpc.encryption_schemes.paillier import Paillier
from tno.mpc.protocols.distributed_keygen import DistributedPaillier

import tno.mpc.protocols.risk_propagation.bank  # to make sphinx find Bank correctly
from .account import Account

logger = logging.getLogger(__name__)
# the expected data type for the nodes
nodes_expected_dtype = np.dtype([("id", np.str_, 100), ("risk_score", np.float64)])


class Accounts(MutableMapping[str, Account]):
    """
    Class containing a dictionary of accounts
    """

    def __init__(self, accounts: set[Account] | None = None) -> None:
        """
        Initializes an Accounts instance

        :param accounts: optional set of accounts to initialise with
        """
        self._dict: dict[str, Account] = (
            {account.label: account for account in accounts}
            if accounts is not None
            else {}
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
    def from_numpy_array(
        cls,
        array: npt.NDArray[np.object_],
        origin: tno.mpc.protocols.risk_propagation.bank.Bank,
        periods: int,
    ) -> Accounts:
        """
        Class method to create an instance of account from a numpy array

        :param array: the numpy array to use in the initialisation with columns "id" (at most 100 unicode characters)
         and "risk_score" (np.float64)
        :param origin: the originating bank
        :param periods: number of periods z considered in the protocol.
        :return: an Accounts instance
        :raise ValueError: raised when provided array does not contain the correct columns
        """
        typed_array = np.rec.array(array, dtype=nodes_expected_dtype)

        logger.info(f"Processing {typed_array.size} accounts")
        accounts = set()
        for account in typed_array:
            accounts.add(
                Account(
                    label=account.id.item(),
                    initial_risk_score=account.risk_score.item(),
                    origin=origin,
                    n_periods=periods,
                )
            )
        return cls(accounts)

    def encrypt(self, public_key: DistributedPaillier | Paillier) -> None:
        """
        Encrypt all accounts

        :param public_key: public key to use in the encryption
        """
        for account in self.values():
            account.encrypt(public_key)

    def get_accounts(
        self, account_keys: set[str] | None = None
    ) -> ValuesView[Account] | Generator[Account]:
        """
        Retrieves all accounts or a subset if account_keys is provided

        :param account_keys: optional set of keys to retreive
        :return: generator of Accounts that are in account_keys
        """
        if account_keys is None:
            return self.values()
        return (self._dict[key] for key in account_keys if key in self.keys())
