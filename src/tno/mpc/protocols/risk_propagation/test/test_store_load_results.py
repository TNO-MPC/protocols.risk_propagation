"""
Compare the result of secure risk propagation with the risk scores when loading intermediate results.
"""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest

from tno.mpc.communication import Pool
from tno.mpc.encryption_schemes.utils import FixedPoint
from tno.mpc.protocols.distributed_keygen import DistributedPaillier

from tno.mpc.protocols.risk_propagation import Player
from tno.mpc.protocols.risk_propagation.test.conftest import get_names_of_pools
from tno.mpc.protocols.risk_propagation.test.test_data.small_multi_period import (
    input_data,
)

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
    results_path: Path | None = None,
) -> Player:
    """
    Returns a party, initialized with a testing dataset.

    :param player_number: the number of this player
    :param http_pool_trio: communication pools that are used
    :param distributed_schemes: distributed paillier schemes
    :param accounts_array: the array of accounts of the player.
    :param transactions_array: the list with the transactions for the periods.
    :param results_path: The path where intermediate results should be stored.
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
        intermediate_results_path=results_path,
    )


async def run_from_intermediate_results(
    iterations: int,
    http_pool_trio: tuple[Pool, Pool, Pool],
    distributed_schemes: tuple[
        DistributedPaillier, DistributedPaillier, DistributedPaillier
    ],
    state_bytes: list[bytes],
) -> dict[str, FixedPoint]:
    """
    Run risk propagation from a previously stored state.

    :param iterations: Number of iterations the risk propagation is performed
    :param http_pool_trio: The pools to use during the test case
    :param distributed_schemes: The distributed paillier schemes used for the secure risk propagation
    :param state_bytes: The bytes containing intermediate results for all the players.
    :return: The final risk scores of all accounts of all parties.

    """

    # Continue secure risk propagation from given state

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

    # Start the parties from the state stored at their respective paths.
    await asyncio.gather(
        *[
            alice.continue_from_stored_results(iterations, state_bytes[0]),
            bob.continue_from_stored_results(iterations, state_bytes[1]),
            charlie.continue_from_stored_results(iterations, state_bytes[2]),
        ]
    )

    return {**alice.risk_scores, **bob.risk_scores, **charlie.risk_scores}


@pytest.mark.asyncio
@pytest.mark.parametrize("iterations", [3])
async def test_store_load_results(
    iterations: int,
    http_pool_trio: tuple[Pool, Pool, Pool],
    distributed_schemes: tuple[
        DistributedPaillier, DistributedPaillier, DistributedPaillier
    ],
) -> None:
    """
    Test which first runs the secure risk propagation and then continues a new player from every intermediate result.
    Tests if these results are always the same.

    :param iterations: Number of iterations the risk propagation is performed
    :param http_pool_trio: The pools to use during the test case
    :param distributed_schemes: The distributed paillier schemes used for the secure risk propagation
    """

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
    secure_risk_scores_init = {
        **alice.risk_scores,
        **bob.risk_scores,
        **charlie.risk_scores,
    }

    alice_states = alice.intermediate_results
    bob_states = bob.intermediate_results
    charlie_states = charlie.intermediate_results
    # try from each intermediate result and gather results
    for period_z in range(len(input_data["numpy_transactions_A"])):
        for iteration in range(iterations):
            # Continue the algorithm from the stored results after this period and iteration.
            current_state = period_z * iterations + iteration
            starting_bytes = [
                alice_states[current_state],
                bob_states[current_state],
                charlie_states[current_state],
            ]
            secure_risk_scores_cont = await run_from_intermediate_results(
                iterations,
                http_pool_trio,
                distributed_schemes,
                starting_bytes,
            )
            # compare these results to original results
            assert secure_risk_scores_cont == secure_risk_scores_init
