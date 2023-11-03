"""
Configuration of a bank
"""
from __future__ import annotations

import asyncio
import logging
from numbers import Integral
from pathlib import Path
from typing import TYPE_CHECKING, Callable

import numpy as np
import numpy.typing as npt

from tno.mpc.communication import Pool, Serialization
from tno.mpc.encryption_schemes.paillier import PaillierCiphertext
from tno.mpc.encryption_schemes.utils import FixedPoint
from tno.mpc.protocols.distributed_keygen import DistributedPaillier

from .bank import Bank
from .exceptions import IncorrectStartException, WrongDeserializationException

if TYPE_CHECKING:
    from .account import (  # prevent cyclic dependency, but do get typing information
        Account,
    )


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
        transactions: npt.NDArray[np.object_] | list[npt.NDArray[np.object_]],
        pool: Pool,
        paillier: DistributedPaillier,
        delta_func: Callable[[int], float] = lambda _: 0.5,
        intermediate_results_path: Path | None = None,
    ):
        """
        Initializes a player instance

        :param name: the name of the player
        :param accounts: an array of accounts containing an initial risk score per account
        :param transactions: an array containing arrays of transactions corresponding to periods
        :param pool: the communication pool to use
        :param paillier: an instance of DistributedPaillier
        :param delta_func: Callable function which uses the iteration index to determine the delta value (must be between [0 and 1)).
        :param intermediate_results_path: the path to the folder where the intermediate results should be stored. If no path is provided, intermediate results are not stored.
        """
        self._total_iterations = 0
        self.name = name
        self._period = 0
        self._iteration = 0
        self._pool = pool
        self._paillier: DistributedPaillier = paillier
        if not isinstance(transactions, list):
            transactions = [transactions]
        self._n_periods = len(transactions)
        self.bank: Bank = Bank(name, n_periods=self._n_periods)
        self._delta_func = delta_func
        self._started_from_intermediate_result = False
        self._intermediate_results: list[bytes] = []
        self._store_intermediate_results = intermediate_results_path is not None
        self._results_path = intermediate_results_path
        self._other_banks: tuple[Bank, ...] = tuple(
            Bank(name, n_periods=self._n_periods)
            for name in self._pool.pool_handlers.keys()
        )

        self._decrypted_scores: dict[str, FixedPoint] | None = None
        self.bank.process_accounts(accounts)

        for bank in self.banks:
            logger.info(f"Processing transactions of bank {bank.name}")
            bank.process_transactions(transactions)

    @property
    def banks(self) -> tuple[Bank, ...]:
        """
        All banks in the protocol

        :return: all banks in the protocol
        """
        return (self.bank,) + self.other_banks

    @property
    def other_banks(self) -> tuple[Bank, ...]:
        """
        The other banks in the protocol

        :return: the other banks in the protocol
        """
        return self._other_banks

    @property
    def risk_scores(self) -> dict[str, FixedPoint]:
        """
        The plaintext risk scores belonging to this player's bank

        :return: plaintext dictionary of risk scores
        :raise AttributeError: raised when risk scores are not available
        """
        if self._decrypted_scores is None:
            raise AttributeError("Risk scores haven't been decrypted (yet)")
        return self._decrypted_scores

    @property
    def intermediate_results(self) -> list[bytes]:
        """
        List of bytes objects of all the intermediate states for this player.

        :return: list of bytes objects.
        :raise AttributeError: raised when there are no intermediate states to return.
        """
        if len(self._intermediate_results) == 0:
            raise AttributeError("No intermediate results have been stored (yet)")
        return self._intermediate_results

    @property
    def delta(self) -> float:
        """
        The delta value for the current iteration

        :return: Float in the range of [0,1)
        :raise AssertionError: raised when the delta is not within the range [0..1)
        """
        delta = self._delta_func(self._iteration)
        assert (
            0 <= delta < 1
        ), f"The delta function provided was not in the range [0..1) for iteration {self._iteration}"
        return delta

    @property
    def results_path(self) -> Path:
        """
        The path object where the intermediate results are stored.

        :return: Path object where the results are stored.
        :raise AttributeError: raised when the results path is not configured for this party.
        """
        if self._results_path is None:
            raise AttributeError("No results path was configured for this player")
        return self._results_path

    def set_current_period(self, period_z: int) -> None:
        """
        Set the period z to be used in the next iteration(s)

        :param period_z: The period z that should be used.
        """
        self._period = period_z
        for bank in self.banks:
            bank.set_current_period(period_z)

    def _compute_new_risk_scores(self) -> dict[str, PaillierCiphertext]:
        """
        Computes new risk scores

        :return: dictionary containing new risk scores
        """
        updated_scores: dict[str, PaillierCiphertext] = {}
        scaled_delta_diff = int(self.COMPENSATION_FACTOR * (1 - self.delta))
        for account_name, account in self.bank.accounts_dict.items():
            updated_scores[account_name] = (
                scaled_delta_diff * account.unsafe_encrypted_risk_score
            )
            if account.total_income != 0:
                scores, total_income = self._get_account_scores_and_total_income(
                    account
                )
                scaled_total_income_recip = (
                    int(self.COMPENSATION_FACTOR * self.delta) // total_income
                )

                for incoming_account, incoming_amount in account.linked_accounts:
                    if incoming_account in scores:
                        weight = scaled_total_income_recip * incoming_amount
                        updated_scores[account_name] += (
                            weight * scores[incoming_account]
                        )
        return updated_scores

    def _get_account_scores_and_total_income(
        self, account: Account
    ) -> tuple[dict[str, PaillierCiphertext], int]:
        """
        Retrieve the risk scores of linked accounts. Accounts that do not have a risk score are not returned.
        The total income for the `account` is adjusted to be the total of all linked accounts with a known risk score.

        :param account: The account of which all linked risk score will be obtained
        :returns: Tuple containing the linked accounts risk scores and the total incoming account of all linked accounts with a known risk score.
        """
        needed_labels = {
            account_label for (account_label, _) in account.linked_accounts
        }

        scores = {
            **self.bank.get_risk_scores(needed_labels),
            **self.bank.get_imported_risk_scores(needed_labels),
        }

        available_labels = set(scores.keys())
        missing_labels = needed_labels.difference(available_labels)

        missing_accounts_amounts = [
            linked_account[1]
            for linked_account in account.linked_accounts
            if linked_account[0] in missing_labels
        ]
        total_missing_accounts_amount = sum(missing_accounts_amounts)
        adjusted_total_income = account.total_income - total_missing_accounts_amount
        return scores, adjusted_total_income

    async def _decrypt_bank(
        self, party: str
    ) -> dict[str, Integral | float | FixedPoint | None]:
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
            # Randomize ciphertexts before sending
            for risk_score in risk_scores.values():
                if not risk_score.fresh:
                    risk_score.randomize()
            await self._pool.broadcast(risk_scores, msg_id=msg_id)
        decrypted_risk_scores: dict[str, Integral | float | FixedPoint | None] = {}
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
        Receive updated risk scores to other banks
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
            f"Awaiting message from {bank.name} for risk scores of period {self._period}, iteration {self._iteration}"
        )
        risk_scores = await self._pool.recv(
            bank.name, msg_id=f"Period {self._period},Iteration {self._iteration}"
        )
        for label, risk_score in risk_scores.items():
            self.bank.set_risk_score(label, risk_score, external=True)

    async def _send_update(self) -> None:
        """
        Sends updated risk scores to other banks
        """
        # Find the accounts of our bank that have an outgoing transaction to another bank, i.e. associated to
        # an account of another bank via a transaction
        all_external_accounts = set()
        for bank in self.other_banks:
            all_external_accounts.update(bank.external_accounts)

        await self.randomize_and_send_updated_risk_scores(all_external_accounts)

    async def randomize_and_send_updated_risk_scores(
        self, all_external_accounts: set[str]
    ) -> None:
        """
        Send updated risk score to the other banks. While ensuring that the ciphertexts are
        randomized only once when it is sent.

        A workaround is used to send randomized ciphertexts to each party. The randomization is done
        once for all parties. In other words all ciphertexts are refreshed if needed at the start
        of this method, but after that they are not rerandomized (which would happen without this
        workaround). Rerandomization would happen, since the communication module marks a ciphertext
        as unfresh when it is sent.

        This is a feature when sending randomizable ciphertext to multiple parties over multiple
        messages.

        Example:
            Let ciphertext `x` be a fresh ciphertext being sent to two parties `A` and `B`. First
            `x` is sent to party `A`. The communication module now sets `x` as unfresh. The
            workaround marks `x` as fresh without randomizing. Now `x` can be sent to party `B`.

        :param all_external_accounts: All account ids, where the account has a transaction to another bank.
        """
        # Randomize the relevant risk scores
        for risk_score in self.bank.get_risk_scores(all_external_accounts).values():
            if not risk_score.fresh:
                risk_score.randomize()

        for bank in self.other_banks:
            for risk_score in self.bank.get_risk_scores(all_external_accounts).values():
                risk_score._fresh = True
            await self._send_updated_risk_scores(bank)

        for risk_score in self.bank.get_risk_scores(all_external_accounts).values():
            risk_score._fresh = False

    async def _send_updated_risk_scores(self, bank: Bank) -> None:
        """
        Sends updated risk score to bank

        :param bank: the bank to send update to
        """
        risk_scores = self.bank.get_risk_scores(bank.external_accounts)

        logger.debug(
            f"Sending message to {bank.name} with {len(risk_scores)} risk scores of period {self._period}, iteration {self._iteration}"
        )
        await self._pool.send(
            bank.name,
            risk_scores,
            msg_id=f"Period {self._period},Iteration {self._iteration}",
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
                self.COMPENSATION_FACTOR**self._total_iterations
            )

    def encrypt_initial_risk_scores(self) -> None:
        """
        Encrypt the initialised risk scores of this player's accounts
        """
        self.bank.encrypt(self._paillier)

    def serialize_current_risk_scores(
        self, current_z: int, current_iteration: int
    ) -> bytes:
        """
        Serialize the risk scores of all accounts of this player.

        :param current_z: The current period z.
        :param current_iteration: The current iteration.
        :return: Bytes storing the risk scores, the current public key, the current period z and the current iteration.
        """

        risk_scores = {k: v._raw_value for k, v in self.bank.get_risk_scores().items()}
        serialized_risk_scores = {
            "risk_scores": risk_scores,
            "public_key": self._paillier.public_key,
            "current_z": current_z,
            "current_iteration": current_iteration,
        }

        packed_risk_scores = Serialization.pack(
            serialized_risk_scores,
            msg_id=f"p{current_z}i{current_iteration}",
            use_pickle=False,
        )

        return packed_risk_scores

    def deserialize_risk_scores(
        self, bytes_object: bytes
    ) -> tuple[dict[str, PaillierCiphertext], int, int]:
        """
        Deserialize the risk scores stored at a given location.

        :param bytes_object: The bytes from which the risk scores should be deserialized.
        :return: Tuple consisting of a dictionary with accounts and risk scores, the period and the iteration in which the loaded risk scores were stored.
        :raise WrongDeserializationException: Raised when the public key of the loaded risk scores or the accounts of the loaded risk scores does not match the current context.
        """

        _, state = Serialization.unpack(bytes_object)
        loaded_public_key = state["public_key"]

        if not self._paillier.public_key == loaded_public_key:
            raise WrongDeserializationException(
                "Paillier public key of loaded ciphertexts does not match the public key currently in use"
            )

        loaded_risk_scores = state["risk_scores"]

        if not self.bank.accounts_dict.keys() == loaded_risk_scores.keys():
            raise WrongDeserializationException(
                "Loaded account keys do not match account keys currently in use"
            )

        formatted_risk_scores: dict[str, PaillierCiphertext] = {}
        for account, score in loaded_risk_scores.items():
            formatted_risk_scores[str(account)] = PaillierCiphertext(
                Serialization.deserialize(score), self._paillier
            )

        return formatted_risk_scores, state["current_z"], state["current_iteration"]

    def intermediate_results_to_disk(
        self, bytes_object: bytes, current_period: int, current_iteration: int
    ) -> None:
        """
        Write the intermediate results to disk.

        :param bytes_object: Bytes representing the current state.
        :param current_period: The current period.
        :param current_iteration: The current iteration.
        """

        with open(
            self.results_path
            / f"{self.name}_results_period_{current_period}_iteration_{current_iteration}.rp",
            "wb",
        ) as file_path:
            file_path.write(bytes_object)

    def intermediate_results_from_disk(self, path: Path) -> bytes:
        """
        Read intermediate results from disk.

        :param path: The path from which the intermediate results should be loaded from disk.
        :return: Bytes containing the intermediate results
        """
        with open(path, "rb") as data_file:
            bytes_object = data_file.read()
        return bytes_object

    async def continue_from_stored_results(
        self, iterations: int, intermediate_results: bytes
    ) -> None:
        """
        Resume the protocol from a stored state.

        :param iterations: the number of iterations to perform in each period.
        :param intermediate_results: the bytes containing the intermediate results to start from.
        """

        loaded_risk_scores, period_z, iteration = self.deserialize_risk_scores(
            intermediate_results
        )

        for account_name, risk_score in loaded_risk_scores.items():
            self.bank.set_risk_score(account_name, risk_score)
        self._started_from_intermediate_result = True
        await self.run_protocol(iterations, period_z, iteration + 1)

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
        self._total_iterations += 1

    async def run_protocol(
        self, iterations: int, starting_period: int = 0, starting_iteration: int = 0
    ) -> None:
        """
        Runs the entire protocol

        :param iterations: the number of iterations to perform in each period
        :param starting_period: the first period that should be run. This is only relevant if the protocol is restarted from intermediate resuls.
        :param starting_iteration: the first iteration that should be performed. This is only relevant if the protocol is restarted from intermediate results.
        :raises IncorrectStartException: Raised if the protocol is incorrectly started.
        """

        if (
            starting_period > 0 or starting_iteration > 0
        ) and not self._started_from_intermediate_result:
            raise IncorrectStartException(
                "Starting period or  starting iteration is set but protocol is not reinstantiated from intermediate state"
            )

        self._paillier.boot_randomness_generation(
            self._calc_required_randomness(
                iterations, starting_period, starting_iteration
            )
        )

        loop = asyncio.get_event_loop()
        if starting_period > 0 or starting_iteration > 0:
            self._total_iterations = starting_period * iterations + starting_iteration
            self._iteration = starting_iteration
        else:
            await loop.run_in_executor(None, self.encrypt_initial_risk_scores)

        for period_z in range(starting_period, self._n_periods):
            self.set_current_period(period_z)
            self._iteration = 0
            for iteration in range(starting_iteration, iterations):
                await self.iteration()
                state_bytes = self.serialize_current_risk_scores(period_z, iteration)
                self._intermediate_results.append(state_bytes)
                if self._store_intermediate_results:
                    self.intermediate_results_to_disk(state_bytes, period_z, iteration)
            starting_iteration = 0

        await self.decrypt()
        self._paillier.shut_down()

    def _calc_required_randomness(
        self, iterations: int, starting_period: int, starting_iteration: int
    ) -> int:
        """
        Calculate the required amount of fresh randomness to run the
        protocol as this play for the given number of iterations.

        :param iterations: The number of iterations that the risk propagation
            will be performed per period.
        :param starting_period: The first period that should be run (only applicable if algorithm is reinstantiated from intermediate state)
        :param starting_iteration: The first iteration that should be run (only applicable if algorithm is reinstantiated from intermediate state)
        :return: The amount of randomness that is required to run the protocol
            for the given number of iterations.
        """
        required_randomness = 0

        n_iterations = iterations - starting_iteration
        # For each period, the set of neighboring nodes changes, and thus we
        # need to recalculate how many risk_scores we must send in that period for each iteration.
        for period_z in range(starting_period, self._n_periods):
            relevant_accounts = set()

            # Only those accounts that are involved in transactions with the other banks need to be sent
            # If the risk score of the same account is sent to multiple banks, it only needs to be randomized once
            for other_bank in self.other_banks:
                other_bank_current_external_transactions = other_bank._all_external_transactions[  # pylint: disable=protected-access
                    period_z
                ]
                bank_external_accounts = {
                    _.sender for _ in other_bank_current_external_transactions
                }
                relevant_accounts.update(bank_external_accounts)

            required_randomness += len(relevant_accounts) * n_iterations
            n_iterations = iterations

        # For decryption, we also need a fresh randomness for each risk score
        # we own, i.e. for each of our accounts.
        required_randomness += len(list(self.bank.accounts_dict))

        return required_randomness

    def update_risk_scores(self) -> None:
        """
        Updates risk scores of all accounts
        """
        updated_scores = self._compute_new_risk_scores()
        for account_name, risk_score in updated_scores.items():
            self.bank.set_risk_score(account_name, risk_score)
