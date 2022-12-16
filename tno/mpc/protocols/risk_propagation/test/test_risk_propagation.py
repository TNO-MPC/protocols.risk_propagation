"""
Module containing tests that can be ran using pytest to test the risk propagation functionality
"""

import asyncio
import logging
from typing import Tuple, cast

import numpy as np
import numpy.typing as npt
import pytest
import pytest_asyncio

from tno.mpc.communication import Pool
from tno.mpc.communication.httphandlers import logger
from tno.mpc.communication.test import (  # pylint: disable=unused-import
    event_loop,
    fixture_pool_http_3p,
)
from tno.mpc.encryption_schemes.utils import FixedPoint
from tno.mpc.protocols.distributed_keygen import DistributedPaillier

from tno.mpc.protocols.risk_propagation import Player

fxp = FixedPoint.fxp
logger.setLevel(logging.WARNING)


@pytest_asyncio.fixture(name="distributed_schemes")
async def fixture_distributed_schemes(
    pool_http_3p: Tuple[Pool, Pool, Pool]
) -> Tuple[DistributedPaillier, DistributedPaillier, DistributedPaillier]:
    """
    Returns a collection of distributed paillier instances

    :param pool_http_3p: communication pools that are used
    :return: distributed paillier instances
    """
    corruption_threshold = 1
    key_length = 256
    prime_threshold = 200
    correct_param_biprime = 20
    stat_sec_shamir = 20
    distributed_schemes = await asyncio.gather(
        *[
            DistributedPaillier.from_security_parameter(
                pool_http_3p[i],
                corruption_threshold,
                key_length,
                prime_threshold,
                correct_param_biprime,
                stat_sec_shamir,
                precision=8,
                distributed=False,
            )
            for i in range(3)
        ]
    )
    return cast(
        Tuple[DistributedPaillier, DistributedPaillier, DistributedPaillier],
        distributed_schemes,
    )


@pytest_asyncio.fixture(name="alice")
async def fixture_alice(
    pool_http_3p: Tuple[Pool, Pool, Pool],
    distributed_schemes: Tuple[
        DistributedPaillier, DistributedPaillier, DistributedPaillier
    ],
) -> Player:
    """
    Returns a party, initalized with a testing dataset.

    :param pool_http_3p: communication pools that are used
    :param distributed_schemes: distributed paillier schemes
    :return: player instance
    """
    alice = list(pool_http_3p[2].pool_handlers)[0]
    bob = list(pool_http_3p[2].pool_handlers)[1]
    charlie = list(pool_http_3p[0].pool_handlers)[1]
    account_datatype = np.dtype([("id", np.unicode_, 100), ("risk_score", np.float64)])
    accounts_array: npt.NDArray[np.object_] = np.array(
        [("a", 0.1), ("b", 0.9), ("c", 0.1)], dtype=account_datatype
    )
    transaction_datatype = np.dtype(
        [
            ("id_source", np.unicode_, 100),
            ("bank_source", np.unicode_, 100),
            ("id_destination", np.unicode_, 100),
            ("bank_destination", np.unicode_, 100),
            ("amount", np.int32),
        ]
    )
    transactions_array: npt.NDArray[np.object_] = np.array(
        [
            ("b", alice, "g", charlie, 100),
            ("b", alice, "a", alice, 100),
            ("d", bob, "c", alice, 100),
        ],
        dtype=transaction_datatype,
    )

    return Player(
        name=alice,
        accounts=accounts_array,
        transactions=transactions_array,
        pool=pool_http_3p[0],
        paillier=distributed_schemes[0],
    )


@pytest_asyncio.fixture(name="bob")
async def fixture_bob(
    pool_http_3p: Tuple[Pool, Pool, Pool],
    distributed_schemes: Tuple[
        DistributedPaillier, DistributedPaillier, DistributedPaillier
    ],
) -> Player:
    """
    Returns a party, initalized with a testing dataset.

    :param pool_http_3p: communication pools that are used
    :param distributed_schemes: distributed paillier schemes
    :return: player instance
    """
    alice = list(pool_http_3p[2].pool_handlers)[0]
    bob = list(pool_http_3p[2].pool_handlers)[1]
    charlie = list(pool_http_3p[0].pool_handlers)[1]
    account_datatype = np.dtype([("id", np.unicode_, 100), ("risk_score", np.float64)])
    accounts_array: npt.NDArray[np.object_] = np.array(
        [("d", 0.7), ("e", 0.8), ("f", 0.0)], dtype=account_datatype
    )
    transaction_datatype = np.dtype(
        [
            ("id_source", np.unicode_, 100),
            ("bank_source", np.unicode_, 100),
            ("id_destination", np.unicode_, 100),
            ("bank_destination", np.unicode_, 100),
            ("amount", np.int32),
        ]
    )
    transactions_array: npt.NDArray[np.object_] = np.array(
        [
            ("d", bob, "c", alice, 100),
            ("e", bob, "g", charlie, 100),
            ("i", charlie, "f", bob, 100),
            ("d", bob, "g", charlie, 100),
        ],
        dtype=transaction_datatype,
    )

    return Player(
        name=bob,
        accounts=accounts_array,
        transactions=transactions_array,
        pool=pool_http_3p[1],
        paillier=distributed_schemes[1],
    )


@pytest_asyncio.fixture(name="charlie")
async def fixture_charlie(
    pool_http_3p: Tuple[Pool, Pool, Pool],
    distributed_schemes: Tuple[
        DistributedPaillier, DistributedPaillier, DistributedPaillier
    ],
) -> Player:
    """
    Returns a party, initalized with a testing dataset.

    :param pool_http_3p: communication pools that are used
    :param distributed_schemes: distributed paillier schemes
    :return: player instance
    """
    alice = list(pool_http_3p[2].pool_handlers)[0]
    bob = list(pool_http_3p[2].pool_handlers)[1]
    charlie = list(pool_http_3p[0].pool_handlers)[1]
    account_datatype = np.dtype([("id", np.unicode_, 100), ("risk_score", np.float64)])
    account_array: npt.NDArray[np.object_] = np.array(
        [("g", 0.0), ("h", 0.0), ("i", 0.2)], dtype=account_datatype
    )

    transaction_datatype = np.dtype(
        [
            ("id_source", np.unicode_, 100),
            ("bank_source", np.unicode_, 100),
            ("id_destination", np.unicode_, 100),
            ("bank_destination", np.unicode_, 100),
            ("amount", np.int32),
        ]
    )
    transactions_array: npt.NDArray[np.object_] = np.array(
        [
            ("e", bob, "g", charlie, 100),
            ("i", charlie, "f", bob, 100),
            ("d", bob, "g", charlie, 100),
            ("b", alice, "g", charlie, 100),
            ("h", charlie, "i", charlie, 100),
            ("g", charlie, "h", charlie, 100),
        ],
        dtype=transaction_datatype,
    )
    return Player(
        name=charlie,
        accounts=account_array,
        transactions=transactions_array,
        pool=pool_http_3p[2],
        paillier=distributed_schemes[2],
    )


@pytest.mark.asyncio
async def test_risk_propagation(
    alice: Player, bob: Player, charlie: Player, iterations: int = 3
) -> None:
    """
    Tests the running of the full protocol

    :param alice: First player
    :param bob: Second player
    :param charlie: Third player
    :param iterations: Number of iterations
    """
    await asyncio.gather(
        *[
            alice.run_protocol(iterations),
            bob.run_protocol(iterations),
            charlie.run_protocol(iterations),
        ]
    )
    correct_outcome = {
        "alice": {"a": fxp(0.35), "b": fxp(0.1125), "c": fxp(0.275)},
        "bob": {"d": fxp(0.0875), "e": fxp(0.1), "f": fxp(0.075)},
        "charlie": {"g": fxp(0.3), "h": fxp(0.3), "i": fxp(0.125)},
    }
    assert alice.risk_scores == correct_outcome["alice"]
    assert bob.risk_scores == correct_outcome["bob"]
    assert charlie.risk_scores == correct_outcome["charlie"]
