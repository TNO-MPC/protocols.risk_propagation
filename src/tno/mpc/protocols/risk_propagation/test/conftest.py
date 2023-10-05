"""
Test fixtures
"""
from __future__ import annotations

import asyncio
from typing import Any, Tuple, cast

import numpy as np
import numpy.typing as npt
import pytest
import pytest_asyncio
from pytest_tno.tno.mpc.communication import determine_pool_scope

from tno.mpc.communication import Pool
from tno.mpc.protocols.distributed_keygen import DistributedPaillier

from tno.mpc.protocols.risk_propagation.accounts import nodes_expected_dtype
from tno.mpc.protocols.risk_propagation.bank import transactions_expected_dtype
from tno.mpc.protocols.risk_propagation.player import Player
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


def get_names_of_pools(pools: tuple[Pool, ...]) -> tuple[str, ...]:
    """
    Get the name of every pool object.

    :param pools: Pools objects.
    :return: Names of the pools.
    """
    # Strictly speaking a pool has no name, but it assigns a name to all its handlers. Since we
    # assign a consistent name to any given http client over all http servers, the names are
    # independent of the pool.

    # A pool does not know its "own name" (again, strictly speaking, it has no name to begin with),
    # so we retrieve the name of the first pool through the second pool.
    return (tuple(pools[1].pool_handlers)[0],) + tuple(pools[0].pool_handlers)


@pytest_asyncio.fixture(name="distributed_schemes", scope=determine_pool_scope)
async def fixture_distributed_schemes(
    http_pool_trio: tuple[Pool, Pool, Pool]
) -> tuple[DistributedPaillier, DistributedPaillier, DistributedPaillier]:
    """
    Returns a collection of distributed paillier instances

    :param http_pool_trio: communication pools that are used
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
                http_pool_trio[_],
                corruption_threshold,
                key_length,
                prime_threshold,
                correct_param_biprime,
                stat_sec_shamir,
                precision=8,
                distributed=False,
                batch_size=1000,
            )
            for _ in range(3)
        ]
    )
    return cast(
        Tuple[DistributedPaillier, DistributedPaillier, DistributedPaillier],
        distributed_schemes,
    )


@pytest_asyncio.fixture(name="alice")
async def fixture_alice(
    http_pool_trio: tuple[Pool, Pool, Pool],
    distributed_schemes: tuple[
        DistributedPaillier, DistributedPaillier, DistributedPaillier
    ],
) -> Player:
    """
    Returns a party, initalized with a testing dataset.

    :param http_pool_trio: communication pools that are used
    :param distributed_schemes: distributed paillier schemes
    :return: player instance
    """
    alice, bob, charlie = get_names_of_pools(http_pool_trio)
    accounts_array: npt.NDArray[np.object_] = np.array(
        [("a", 0.1), ("b", 0.9), ("c", 0.1)], dtype=nodes_expected_dtype
    )
    transactions_array: npt.NDArray[np.object_] = np.array(
        [
            ("b", alice, "g", charlie, 100),
            ("b", alice, "a", alice, 100),
            ("d", bob, "c", alice, 100),
        ],
        dtype=transactions_expected_dtype,
    )

    return Player(
        name=alice,
        accounts=accounts_array,
        transactions=transactions_array,
        pool=http_pool_trio[0],
        paillier=distributed_schemes[0],
    )


@pytest_asyncio.fixture(name="bob")
async def fixture_bob(
    http_pool_trio: tuple[Pool, Pool, Pool],
    distributed_schemes: tuple[
        DistributedPaillier, DistributedPaillier, DistributedPaillier
    ],
) -> Player:
    """
    Returns a party, initalized with a testing dataset.

    :param http_pool_trio: communication pools that are used
    :param distributed_schemes: distributed paillier schemes
    :return: player instance
    """
    alice, bob, charlie = get_names_of_pools(http_pool_trio)
    accounts_array: npt.NDArray[np.object_] = np.array(
        [("d", 0.7), ("e", 0.8), ("f", 0.0)], dtype=nodes_expected_dtype
    )
    transactions_array: npt.NDArray[np.object_] = np.array(
        [
            ("d", bob, "c", alice, 100),
            ("e", bob, "g", charlie, 100),
            ("i", charlie, "f", bob, 100),
            ("d", bob, "g", charlie, 100),
        ],
        dtype=transactions_expected_dtype,
    )

    return Player(
        name=bob,
        accounts=accounts_array,
        transactions=transactions_array,
        pool=http_pool_trio[1],
        paillier=distributed_schemes[1],
    )


@pytest_asyncio.fixture(name="charlie")
async def fixture_charlie(
    http_pool_trio: tuple[Pool, Pool, Pool],
    distributed_schemes: tuple[
        DistributedPaillier, DistributedPaillier, DistributedPaillier
    ],
) -> Player:
    """
    Returns a party, initalized with a testing dataset.

    :param http_pool_trio: communication pools that are used
    :param distributed_schemes: distributed paillier schemes
    :return: player instance
    """
    alice, bob, charlie = get_names_of_pools(http_pool_trio)
    account_datatype = np.dtype([("id", np.unicode_, 100), ("risk_score", np.float64)])
    account_array: npt.NDArray[np.object_] = np.array(
        [("g", 0.0), ("h", 0.0), ("i", 0.2)], dtype=account_datatype
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
        dtype=transactions_expected_dtype,
    )
    return Player(
        name=charlie,
        accounts=account_array,
        transactions=transactions_array,
        pool=http_pool_trio[2],
        paillier=distributed_schemes[2],
    )
