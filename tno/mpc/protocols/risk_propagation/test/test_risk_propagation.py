"""
Module containing tests that can be ran using pytest to test the risk propagation functionality
"""

import asyncio
import logging
from typing import Tuple, cast

import pandas as pd
import pytest

from tno.mpc.communication import Pool
from tno.mpc.communication.httphandlers import logger
from tno.mpc.communication.test import (  # pylint: disable=unused-import
    fixture_pool_http_3p,
)
from tno.mpc.encryption_schemes.utils import FixedPoint
from tno.mpc.protocols.distributed_keygen import DistributedPaillier

from tno.mpc.protocols.risk_propagation import Player

fxp = FixedPoint.fxp
logger.setLevel(logging.WARNING)


@pytest.fixture(name="distributed_schemes")
@pytest.mark.asyncio
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


@pytest.fixture(name="alice")
@pytest.mark.asyncio
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
    accounts_df = pd.DataFrame({"id": ["a", "b", "c"], "risk_score": [0.1, 0.9, 0.1]})
    transactions_df = pd.DataFrame(
        {
            "id_source": ["b", "b", "d"],
            "bank_source": [alice, alice, bob],
            "id_destination": ["g", "a", "c"],
            "bank_destination": [charlie, alice, alice],
            "amount": [100, 100, 100],
        }
    )
    return Player(
        name=alice,
        accounts=accounts_df,
        transactions=transactions_df,
        pool=pool_http_3p[0],
        paillier=distributed_schemes[0],
    )


@pytest.fixture(name="bob")
@pytest.mark.asyncio
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
    accounts_df = pd.DataFrame({"id": ["d", "e", "f"], "risk_score": [0.7, 0.8, 0.0]})
    transactions_df = pd.DataFrame(
        {
            "id_source": ["d", "e", "i", "d"],
            "bank_source": [bob, bob, charlie, bob],
            "id_destination": ["c", "g", "f", "g"],
            "bank_destination": [alice, charlie, bob, charlie],
            "amount": [100, 100, 100, 100],
        }
    )
    return Player(
        name=bob,
        accounts=accounts_df,
        transactions=transactions_df,
        pool=pool_http_3p[1],
        paillier=distributed_schemes[1],
    )


@pytest.fixture(name="charlie")
@pytest.mark.asyncio
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
    accounts_df = pd.DataFrame({"id": ["g", "h", "i"], "risk_score": [0.0, 0.0, 0.2]})

    transactions_df = pd.DataFrame(
        {
            "id_source": ["e", "i", "d", "b", "h", "g"],
            "bank_source": [bob, charlie, bob, alice, charlie, charlie],
            "id_destination": ["g", "f", "g", "g", "i", "h"],
            "bank_destination": [
                charlie,
                bob,
                charlie,
                charlie,
                charlie,
                charlie,
            ],
            "amount": [100, 100, 100, 100, 100, 100],
        }
    )
    return Player(
        name=charlie,
        accounts=accounts_df,
        transactions=transactions_df,
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
