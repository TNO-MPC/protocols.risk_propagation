"""
Compare the result of plaintext risk propagation with the secure risk propagation.
"""

import asyncio
import logging
from typing import Any, List, Tuple, Union, cast

import numpy as np
import numpy.typing as npt
import pytest
import pytest_asyncio

from tno.mpc.communication import Pool
from tno.mpc.communication.test import (  # pylint: disable=unused-import
    event_loop,
    fixture_pool_http_3p,
)
from tno.mpc.encryption_schemes.utils import FixedPoint
from tno.mpc.protocols.distributed_keygen import DistributedPaillier

from tno.mpc.protocols.risk_propagation import Player
from tno.mpc.protocols.risk_propagation.test import plaintext_risk_propagation
from tno.mpc.protocols.risk_propagation.test.test_data import InputData
from tno.mpc.protocols.risk_propagation.test.test_data.missing_nodes import (
    input_data as MissingNodes,
)
from tno.mpc.protocols.risk_propagation.test.test_data.small import (
    input_data as SmallSinglePeriod,
)
from tno.mpc.protocols.risk_propagation.test.test_data.small_multi_period import (
    input_data as SmallMultiPeriod,
)
from tno.mpc.protocols.risk_propagation.test.test_data.small_multi_period_test import (
    input_data as SmallMultiPeriodTest,
)

logging.getLogger("tno.mpc.communication.httphandlers").setLevel("CRITICAL")
logging.getLogger("aiohttp.access").setLevel("CRITICAL")

fxp = FixedPoint.fxp


async def create_player(
    player_number: int,
    pool_http_3p: Tuple[Pool, Pool, Pool],
    distributed_schemes: Tuple[
        DistributedPaillier, DistributedPaillier, DistributedPaillier
    ],
    accounts_array: npt.NDArray[np.object_],
    transactions_array: Union[npt.NDArray[np.object_], List[npt.NDArray[np.object_]]],
) -> Player:
    """
    Returns a party, initialized with a testing dataset.

    :param player_number: the number of this player
    :param pool_http_3p: communication pools that are used
    :param distributed_schemes: distributed paillier schemes
    :param accounts_array: the array of accounts of the player.
    :param transactions_array: the list with the transactions for the periods.
    :return: player instance
    """
    alice = list(pool_http_3p[2].pool_handlers)[0]
    bob = list(pool_http_3p[2].pool_handlers)[1]
    charlie = list(pool_http_3p[0].pool_handlers)[1]
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
        pool=pool_http_3p[player_number],
        paillier=distributed_schemes[player_number],
    )


@pytest.fixture(
    name="input_data",
    params=[MissingNodes, SmallMultiPeriod, SmallSinglePeriod, SmallMultiPeriodTest],
    ids=[
        "missing-objective-indicators",
        "multi-period",
        "single-period",
        "multi-period-test",
    ],
)
def fixture_input_data(request: Any) -> InputData:
    """
    Create a fixture returning the typed dict `InputData`.

    :param request: Parameter containing information from the pytest fixture annotation
    :return: The input data for a test case
    """
    data_set: InputData = request.param
    return data_set


@pytest_asyncio.fixture(name="distributed_schemes", scope="module")
async def fixture_distributed_schemes(
    pool_http_3p: Tuple[Pool, Pool, Pool]
) -> Tuple[DistributedPaillier, DistributedPaillier, DistributedPaillier]:
    """
    Returns a collection of distributed paillier instances

    :param pool_http_3p: communication pools that are used
    :return: distributed paillier instances
    """
    corruption_threshold = 1
    key_length = 640
    prime_threshold = 200
    correct_param_biprime = 20
    stat_sec_shamir = 20
    distributed_schemes = await asyncio.gather(
        *[
            DistributedPaillier.from_security_parameter(
                pool_http_3p[_],
                corruption_threshold,
                key_length,
                prime_threshold,
                correct_param_biprime,
                stat_sec_shamir,
                precision=8,
                distributed=False,
            )
            for _ in range(3)
        ]
    )
    return cast(
        Tuple[DistributedPaillier, DistributedPaillier, DistributedPaillier],
        distributed_schemes,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("iterations", [0, 1, 2, 3])
async def test_compare_plaintext(
    iterations: int,
    input_data: InputData,
    pool_http_3p: Tuple[Pool, Pool, Pool],
    distributed_schemes: Tuple[
        DistributedPaillier, DistributedPaillier, DistributedPaillier
    ],
) -> None:
    """
    Test case which first runs the plaintext risk propagation and then the secure risk propagation. The results of both
    are compared and at most `1e-5` error is allowed.

    :param iterations: Number of iterations the risk propagation is performed
    :param input_data: The input data for the test case
    :param pool_http_3p: The pools to use during the test case
    :param distributed_schemes: The distributed paillier schemes used for the secure risk propagation
    """
    plaintext_risk_scores, _ = plaintext_risk_propagation.risk_propagation(
        input_data["nodes_dict"],
        input_data["transactions_dict"],
        input_data["transaction_totals_dict"],
        lambda x: 0.5,
        iterations,
    )

    # create players
    alice = await create_player(
        0,
        pool_http_3p,
        distributed_schemes,
        input_data["numpy_nodes_A"],
        input_data["numpy_transactions_A"],
    )
    bob = await create_player(
        1,
        pool_http_3p,
        distributed_schemes,
        input_data["numpy_nodes_B"],
        input_data["numpy_transactions_B"],
    )
    charlie = await create_player(
        2,
        pool_http_3p,
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
