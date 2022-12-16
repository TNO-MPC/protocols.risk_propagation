"""
Configuration of a bank
"""
import asyncio
import logging
from numbers import Integral
from typing import Dict, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt

from tno.mpc.communication import Pool
from tno.mpc.encryption_schemes.paillier import PaillierCiphertext
from tno.mpc.encryption_schemes.utils import FixedPoint
from tno.mpc.protocols.distributed_keygen import DistributedPaillier

from .bank import Bank

logger = logging.getLogger(__name__)


class Player:
    """
    Player class performing steps in protocol
    """

    # This factor is used to make sure that the computation of the risk scores
    # remains accurate enough (we have to compensate for scalar division)
    COMPENSATION_FACTOR = 10**12

    def __init__(
        self,
        name: str,
        accounts: npt.NDArray[np.object_],
        transactions: npt.NDArray[np.object_],
        pool: Pool,
        paillier: DistributedPaillier,
        delta: float = 0.5,
    ):
        """
        Initializes a player instance

        :param name: the name of the player
        :param accounts: an array of accounts containing an initial risk score per account
        :param transactions: an array of transactions
        :param pool: the communication pool to use
        :param paillier: an instance of DistributedPaillier
        :param delta: the delta to use
        """
        self._iteration = 0
        self._pool = pool
        self._paillier = paillier
        self.bank: Bank = Bank(name)
        self._other_banks: Tuple[Bank, ...] = tuple(
            Bank(name) for name in self._pool.pool_handlers.keys()
        )

        self._decrypted_scores: Optional[Dict[str, FixedPoint]] = None
        self.bank.process_accounts(accounts, delta)

        for bank in self.banks:
            logger.info(f"Processing transactions of bank {bank.name}")
            bank.process_transactions(transactions)

    @property
    def banks(self) -> Tuple[Bank, ...]:
        """
        All banks in the protocol

        :return: all banks in the protocol
        """
        return (self.bank,) + self.other_banks

    @property
    def other_banks(self) -> Tuple[Bank, ...]:
        """
        The other banks in the protocol

        :return: the other banks in the protocol
        """
        return self._other_banks

    @property
    def risk_scores(self) -> Dict[str, FixedPoint]:
        """
        The plaintext risk scores belonging to this player's bank

        :return: plaintext dictionary of risk scores
        :raise AttributeError: raised when risk scores are not available
        """
        if self._decrypted_scores is None:
            raise AttributeError("Risk scores haven been decrypted (yet)")
        return self._decrypted_scores

    def _compute_new_risk_scores(self) -> Dict[str, PaillierCiphertext]:
        """
        Computes new risk scores

        :return: dictionary containing new risk scores
        """
        scores = self.bank.risk_scores
        updated_scores: Dict[str, Union[PaillierCiphertext]] = {}
        for account_name, account in self.bank.accounts_dict.items():
            scaled_delta_diff = int(self.COMPENSATION_FACTOR * (1 - account.delta))
            updated_scores[account_name] = scaled_delta_diff * account.safe_risk_score
            if account.total_income != 0:
                scaled_total_income_recip = (
                    int(self.COMPENSATION_FACTOR * account.delta)
                    // account.total_income
                )
                for incoming_account, incoming_amount in account.linked_accounts:
                    weight = scaled_total_income_recip * incoming_amount
                    updated_scores[account_name] += weight * scores[incoming_account]
        return updated_scores

    async def _decrypt_bank(
        self, party: str
    ) -> Dict[str, Union[Integral, float, FixedPoint, None]]:
        """
        Decrypts the risk scores of party and reveals them to party

        :param party: the party to decrypt
        :return: a dictionary of decrypted risk scores
        """
        if party in (_.name for _ in self.other_banks):
            msg_id = f"Decryption {party}"
            logger.debug(f"Awaiting message with id {msg_id}")
            risk_scores = await self._pool.recv(party, msg_id)
        else:
            risk_scores = self.bank.get_risk_scores()
            msg_id = f"Decryption {self.bank.name}"
            logger.debug(f"Sending message with id: {msg_id}")
            await self._pool.broadcast(risk_scores, msg_id=msg_id)
        decrypted_risk_scores: Dict[str, Union[Integral, float, FixedPoint, None]] = {}
        risk_score_mapping = {}
        risk_score_values = []
        for index, (key, value) in enumerate(risk_scores.items()):
            risk_score_mapping[key] = index
            risk_score_values.append(value)

        logger.info(f"Awaiting paillier decryption results for party: {party}")
        decrypted_risk_score_list = await self._paillier.decrypt_sequence(
            risk_score_values, receivers=[party]
        )

        for key in risk_scores.keys():
            decrypted_risk_scores[key] = (
                decrypted_risk_score_list[risk_score_mapping[key]]
                if decrypted_risk_score_list is not None
                else None
            )
        return decrypted_risk_scores

    async def _receive_update(self) -> None:
        """
        Sends updated risk scores to other banks
        """
        await asyncio.gather(
            *[self._receive_updated_risk_scores(bank) for bank in self.other_banks]
        )

    async def _receive_updated_risk_scores(self, bank: Bank) -> None:
        """
        Receives updated the scores of the external nodes of bank

        :param bank: the bank to update
        """
        logger.debug(
            f"Awaiting message from {bank.name} for risk scores of iteration {self._iteration}"
        )
        risk_scores = await self._pool.recv(
            bank.name, msg_id=f"Iteration {self._iteration}"
        )
        for label, risk_score in risk_scores.items():
            self.bank.set_risk_score(label, risk_score, external=True)

    async def _send_update(self) -> None:
        """
        Sends updated risk scores to other banks
        """
        await asyncio.gather(
            *[self._send_updated_risk_scores(bank) for bank in self.other_banks]
        )

    async def _send_updated_risk_scores(self, bank: Bank) -> None:
        """
        Sends updated risk score to bank

        :param bank: the bank to send update to
        """
        risk_scores = self.bank.get_risk_scores(bank.external_accounts)

        logger.debug(
            f"Sending message to {bank.name} with {len(risk_scores)} risk scores of iteration {self._iteration}"
        )
        await self._pool.send(
            bank.name, risk_scores, msg_id=f"Iteration {self._iteration}"
        )

    async def decrypt(self) -> None:
        """
        Decryption of the risk scores per bank, compensating for the COMPENSATION_FACTOR
        """
        for party in self._paillier.party_indices.keys():
            if party == "self":
                decrypted_scores = await self._decrypt_bank(party)
            else:
                await self._decrypt_bank(party)
        self._decrypted_scores = {}
        for account, scaled_score in decrypted_scores.items():
            self._decrypted_scores[account] = scaled_score / (
                self.COMPENSATION_FACTOR**self._iteration
            )

    def encrypt_initial_risk_scores(self) -> None:
        """
        Encrypt the initialised risk scores of this player's accounts
        """
        self.bank.encrypt(self._paillier)

    async def iteration(self) -> None:
        """
        Perform a single iteration
        """
        await asyncio.gather(
            *[
                self._send_update(),
                self._receive_update(),
            ]
        )
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self.update_risk_scores)
        self._iteration += 1

    async def run_protocol(self, iterations: int) -> None:
        """
        Runs the entire protocol

        :param iterations: the number of iterations to perform
        """
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self.encrypt_initial_risk_scores)
        for _ in range(iterations):
            await self.iteration()
        await self.decrypt()

    def update_risk_scores(self) -> None:
        """
        Updates risk scores of all accounts
        """
        updated_scores = self._compute_new_risk_scores()
        for account_name, risk_score in updated_scores.items():
            self.bank.set_risk_score(account_name, risk_score)
