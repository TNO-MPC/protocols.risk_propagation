"""
Compare the result of plaintext risk propagation with the secure risk propagation.
"""
from __future__ import annotations

import asyncio
import logging

import numpy as np
import numpy.typing as npt
import pytest

from tno.mpc.communication import Pool
from tno.mpc.encryption_schemes.utils import FixedPoint
from tno.mpc.protocols.distributed_keygen import DistributedPaillier

from tno.mpc.protocols.risk_propagation import Player
from tno.mpc.protocols.risk_propagation.test import plaintext_risk_propagation
from tno.mpc.protocols.risk_propagation.test.conftest import get_names_of_pools
from tno.mpc.protocols.risk_propagation.test.test_data import InputData

logging.getLogger("tno.mpc.communication.httphandlers").setLevel("CRITICAL")
logging.getLogger("aiohttp.access").setLevel("CRITICAL")

fxp = FixedPoint.fxp


async def create_player(
    player_number: int,
    http_pool_trio: tuple[Pool, Pool, Pool],
    distributed_schemes: tuple[
        DistributedPaillier, DistributedPaillier, DistributedPaillier
    ],
    accounts_array: npt.NDArray[np.object_],
    transactions_array: npt.NDArray[np.object_] | list[npt.NDArray[np.object_]],
) -> Player:
    """
    Returns a party, initialized with a testing dataset.

    :param player_number: the number of this player
    :param http_pool_trio: communication pools that are used
    :param distributed_schemes: distributed paillier schemes
    :param accounts_array: the array of accounts of the player.
    :param transactions_array: the list with the transactions for the periods.
    :return: player instance
    """
    alice, bob, charlie = get_names_of_pools(http_pool_trio)
    if player_number == 0:
        player_name = alice
    elif player_number == 1:
        player_name = bob
    else:
        player_name = charlie

    return Player(
        name=player_name,
        accounts=accounts_array,
        transactions=transactions_array,
        pool=http_pool_trio[player_number],
        paillier=distributed_schemes[player_number],
        delta_func=lambda i: float(0.3 / (i + 1) ** 0.8),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("iterations", [0, 1, 2, 3])
async def test_compare_plaintext(
    iterations: int,
    input_data: InputData,
    http_pool_trio: tuple[Pool, Pool, Pool],
    distributed_schemes: tuple[
        DistributedPaillier, DistributedPaillier, DistributedPaillier
    ],
) -> None:
    """
    Test case which first runs the plaintext risk propagation and then the secure risk propagation. The results of both
    are compared and at most `1e-5` error is allowed.

    :param iterations: Number of iterations the risk propagation is performed
    :param input_data: The input data for the test case
    :param http_pool_trio: The pools to use during the test case
    :param distributed_schemes: The distributed paillier schemes used for the secure risk propagation
    """
    plaintext_risk_scores, _ = plaintext_risk_propagation.risk_propagation(
        input_data["nodes_dict"],
        input_data["transactions_dict"],
        input_data["transaction_totals_dict"],
        lambda i: float(0.3 / i**0.8),
        iterations,
    )

    # create players
    alice = await create_player(
        0,
        http_pool_trio,
        distributed_schemes,
        input_data["numpy_nodes_A"],
        input_data["numpy_transactions_A"],
    )
    bob = await create_player(
        1,
        http_pool_trio,
        distributed_schemes,
        input_data["numpy_nodes_B"],
        input_data["numpy_transactions_B"],
    )
    charlie = await create_player(
        2,
        http_pool_trio,
        distributed_schemes,
        input_data["numpy_nodes_C"],
        input_data["numpy_transactions_C"],
    )

    # run secure risk propagation
    await asyncio.gather(
        *[
            alice.run_protocol(iterations),
            bob.run_protocol(iterations),
            charlie.run_protocol(iterations),
        ]
    )

    # combine results
    secure_risk_scores = {**alice.risk_scores, **bob.risk_scores, **charlie.risk_scores}

    # transform plaintext risk scores to fixed points
    fixed_point_plaintext_risk_scores = {
        str(key): fxp(value) for key, value in plaintext_risk_scores.items()
    }

    # compare both results
    for key in fixed_point_plaintext_risk_scores:
        assert (
            abs(fixed_point_plaintext_risk_scores[key] - secure_risk_scores[key]) < 1e-5
        )
