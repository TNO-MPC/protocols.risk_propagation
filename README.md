# TNO MPC Lab - Protocols - Secure Risk Propagation

The TNO MPC lab consists of generic software components, procedures, and functionalities developed and maintained on a regular basis to facilitate and aid in the development of MPC solutions. The lab is a cross-project initiative allowing us to integrate and reuse previously developed MPC functionalities to boost the development of new protocols and solutions.

The package tno.mpc.protocols.risk_propagation is part of the TNO Python Toolbox.

The research activities that led to this protocol and implementation were funded by ABN AMRO, Rabobank, PPS-surcharge for Research and Innovation of the Dutch Ministry of Economic Affairs and Climate Policy, and TNO's Appl.AI programme.

*Limitations in (end-)use: the content of this software package may solely be used for applications that comply with international export control laws.*

## Documentation

Documentation of the tno.mpc.protocols.risk_propagation package can be found [here](https://docs.mpc.tno.nl/protocols/risk_propagation/0.1.6).

## Install

Easily install the tno.mpc.protocols.risk_propagation package using pip:
```console
$ python -m pip install tno.mpc.protocols.risk_propagation
```

If you wish to run the tests you can use:
```console
$ python -m pip install 'tno.mpc.protocols.risk_propagation[tests]'
```

### Note:
A significant performance improvement can be achieved by installing the GMPY2 library.
```console
$ python -m pip install 'tno.mpc.protocols.risk_propagation[gmpy]'
```

## Protocol description
Risk propagation is an algorithm that propagates risk scores through a (transaction) network.
Using distributed partial homomorphic encryption, the secure risk propagation implementation performs this algorithm on a network that is distributed among multiple parties.
As input of the algorithm, every party possesses a list of nodes (i.e. bank accounts) with initial risk scores and has a list of transactions (weighted, directed edges) from and to its bank accounts.
Using encrypted incoming risk scores scores from other parties, every party can securely update its risk scores using the formula for risk propagation.
After a set number of iterations, the eventual risk scores are revealed to the parties that own the accounts, using the distributed private key.

Figure 1 demonstrates a high-level overview of the idea behind the protocol.
<figure>
  <img src="https://raw.githubusercontent.com/TNO-MPC/protocols.risk_propagation/main/assets/risk-propagation-overview.svg" width=100% alt="Risk Propagation High Level Overview"/>
  <figcaption>

__Figure 1.__ _We consider a three-bank scenario (Orange, Blue, and Purple). In this scenario the first (left) account at bank Orange is classified as high risk (due to e.g., large cash deposits) by bank Orange.
This account wishes to launder its resources. To stay under the radar, the resources are funnelled through multiple accounts, at various banks, before arriving at their eventual destination, e.g., the account at bank Purple (right).
To detect money laundering, we wish to follow (propagate) the risky money and classify the endpoint as high risk too. Full (global) knowledge of the network enables us to propagate the risk.
However, how can we achieve something similar when there is only partial (local) knowledge of the entire network available? This is where MPC comes into play._
  </figcaption>
</figure>

## Usage

The protocol is symmetric.

>`example_usage.py`
>```python
>"""
>    Example usage for performing secure risk propagation
>    Run three separate instances e.g.,
>    $ python example_usage.py -p Alice
>    $ python example_usage.py -p Bob
>    $ python example_usage.py -p Charlie
>"""
>import argparse
>import asyncio
>
>from tno.mpc.communication import Pool
>from tno.mpc.protocols.distributed_keygen import DistributedPaillier
>
>from tno.mpc.protocols.risk_propagation import Player
>
>"""
>Default parameters for distributed keygen
>"""
>corruption_threshold = 1  # corruption threshold
>key_length = 256  # bit length of private key
>prime_thresh = 2000  # threshold for primality check
>correct_param_biprime = 100  # correctness parameter for biprimality test
>stat_sec_shamir = (
>    40  # statistical security parameter for secret sharing over the integers
>)
>
>
>def parse_args():
>    parser = argparse.ArgumentParser()
>    parser.add_argument(
>        "-p",
>        "--player",
>        help="Name of the sending player",
>        type=str.lower,
>        required=True,
>        choices=["alice", "bob", "charlie", "all"],
>    )
>    args = parser.parse_args()
>    return args
>
>
>async def main(player, pool, nodes, transactions, distributed):
>    distributed_paillier = await DistributedPaillier.from_security_parameter(
>        pool,
>        corruption_threshold,
>        key_length,
>        prime_thresh,
>        correct_param_biprime,
>        stat_sec_shamir,
>        precision=8,
>        distributed=distributed,
>    )
>    player_instance = Player(player, nodes, transactions, pool, distributed_paillier)
>
>    await player_instance.run_protocol(iterations=3)
>    print(player_instance.risk_scores)
>    await asyncio.gather(
>        *[shutdown(pool, player) for player in pool.pool_handlers.keys()]
>    )
>
>
>async def shutdown(pool, player):
>    await pool.send(player, "Shutting down..")
>    return await pool.recv(player)
>
>
>async def generate_instance(player, distributed=True):
>    parties = {
>        "alice": {"address": "127.0.0.1", "port": 8080},
>        "bob": {"address": "127.0.0.1", "port": 8081},
>        "charlie": {"address": "127.0.0.1", "port": 8082},
>    }
>
>    port = parties[player]["port"]
>    del parties[player]
>
>    pool = Pool()
>    pool.add_http_server(port=port)
>    for name, party in parties.items():
>        assert "address" in party
>        pool.add_http_client(
>            name, party["address"], port=party["port"] if "port" in party else 80
>        )  # default port=80
>
>    if player == "alice":
>        from tno.mpc.protocols.risk_propagation.example_data import nodes_A as nodes
>        from tno.mpc.protocols.risk_propagation.example_data import (
>            transactions_A as transactions,
>        )
>    elif player == "bob":
>        from tno.mpc.protocols.risk_propagation.example_data import nodes_B as nodes
>        from tno.mpc.protocols.risk_propagation.example_data import (
>            transactions_B as transactions,
>        )
>    elif player == "charlie":
>        from tno.mpc.protocols.risk_propagation.example_data import nodes_C as nodes
>        from tno.mpc.protocols.risk_propagation.example_data import (
>            transactions_C as transactions,
>        )
>
>    await main(player, pool, nodes, transactions, distributed)
>
>
>async def all():
>    await asyncio.gather(
>        *[
>            generate_instance("alice", distributed=False),
>            generate_instance("bob", distributed=False),
>            generate_instance("charlie", distributed=False),
>        ],
>        return_exceptions=True,
>    )
>
>
>if __name__ == "__main__":
>    # Parse arguments and acquire configuration parameters
>    args = parse_args()
>    player = args.player
>    loop = asyncio.get_event_loop()
>    if player == "all":
>        loop.run_until_complete(all())
>    else:
>        loop.run_until_complete(generate_instance(player))
>```

- Run three separate instances specifying the players:
  ```console
  $ python example_usage.py -p Alice
  $ python example_usage.py -p Bob
  $ python example_usage.py -p Charlie
  ```
or
- Run all three players in one Python instance:
  ```console
  $ python example_usage.py -p all
  ```
