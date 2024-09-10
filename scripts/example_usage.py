#!/usr/bin/python3
"""
    Example usage for performing secure risk propagation
    Run three separate instances e.g.,
    $ python example_usage.py -p Alice
    $ python example_usage.py -p Bob
    $ python example_usage.py -p Charlie
"""
import argparse
import asyncio

from tno.mpc.communication import Pool
from tno.mpc.communication.httphandlers import logger
from tno.mpc.protocols.distributed_keygen import DistributedPaillier

from tno.mpc.protocols.risk_propagation import Player

"""
Default parameters for distributed keygen
"""
corruption_threshold = 1  # corruption threshold
key_length = 256  # bit length of private key
prime_thresh = 2000  # threshold for primality check
correct_param_biprime = 100  # correctness parameter for biprimality test
stat_sec_shamir = (
    40  # statistical security parameter for secret sharing over the integers
)

logger.setLevel("CRITICAL")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p",
        "--player",
        help="Name of the sending player",
        type=str.lower,
        required=True,
        choices=["alice", "bob", "charlie", "all"],
    )
    args = parser.parse_args()
    return args


async def main(player, pool, nodes, transactions, distributed):
    distributed_paillier = await DistributedPaillier.from_security_parameter(
        pool,
        corruption_threshold,
        key_length,
        prime_thresh,
        correct_param_biprime,
        stat_sec_shamir,
        precision=8,
        distributed=distributed,
    )
    player_instance = Player(player, nodes, transactions, pool, distributed_paillier)

    await player_instance.run_protocol(iterations=3)
    print(player_instance.risk_scores)
    await asyncio.gather(
        *[shutdown(pool, player) for player in pool.pool_handlers.keys()]
    )


async def shutdown(pool, player):
    await pool.send(player, "Shutting down..")
    return await pool.recv(player)


async def generate_instance(player, distributed=True):
    parties = {
        "alice": {"address": "127.0.0.1", "port": 8080},
        "bob": {"address": "127.0.0.1", "port": 8081},
        "charlie": {"address": "127.0.0.1", "port": 8082},
    }

    port = parties[player]["port"]
    del parties[player]

    pool = Pool()
    pool.add_http_server(port=port)
    for name, party in parties.items():
        assert "address" in party
        pool.add_http_client(
            name, party["address"], port=party["port"] if "port" in party else 80
        )  # default port=80

    if player == "alice":
        from tno.mpc.protocols.risk_propagation.example_data import nodes_A as nodes
        from tno.mpc.protocols.risk_propagation.example_data import (
            transactions_A as transactions,
        )
    elif player == "bob":
        from tno.mpc.protocols.risk_propagation.example_data import nodes_B as nodes
        from tno.mpc.protocols.risk_propagation.example_data import (
            transactions_B as transactions,
        )
    elif player == "charlie":
        from tno.mpc.protocols.risk_propagation.example_data import nodes_C as nodes
        from tno.mpc.protocols.risk_propagation.example_data import (
            transactions_C as transactions,
        )

    await main(player, pool, nodes, transactions, distributed)


async def all():
    await asyncio.gather(
        *[
            generate_instance("alice", distributed=False),
            generate_instance("bob", distributed=False),
            generate_instance("charlie", distributed=False),
        ],
        return_exceptions=True,
    )


if __name__ == "__main__":
    # Parse arguments and acquire configuration parameters
    args = parse_args()
    player = args.player
    loop = asyncio.get_event_loop()
    if player == "all":
        loop.run_until_complete(all())
    else:
        loop.run_until_complete(generate_instance(player))
