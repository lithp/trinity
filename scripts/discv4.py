#!/usr/bin/env python
"""
Attempts to iterate over the entire DHT.
It starts out only knowing about the bootnodes.
For every node:
- try to bond with that node
- if the bond succeeds, send a FIND_NODE packet for a randomly chosen nodeid
- add all neighbours we haven't seen before to the list of nodes

While this script generates a lot of network traffic it does not send more than one
FIND_NODE packet to each node that it bonds with. That seems to be enough to find ~7800
nodes, and shouldn't place too large a load upon the network.

Known bugs:
- this script does not automatically quit when there is no more work to be done
"""

import argparse
import logging
import random
import math

import trio

from async_service import TrioManager

from p2p.abc import NodeAPI
from p2p import ecies, kademlia, constants

from p2p.discovery import (
    DiscoveryProtocol,
)


logger = logging.getLogger('discv4')



def random_nodeid() -> int:
    return random.randint(0, constants.KADEMLIA_MAX_NODE_ID)


async def crawl(protocol, bootnodes) -> None:
    """
    Attempt to traverse the network. All found (and successfully bonded) nodes are logged.
    """

    unbonded_send, unbonded_recv = trio.open_memory_channel(math.inf)
    queued_nodes: Set[NodeAPI] = set()

    bonded_nodes = set()

    async def enqueue_node(remote):
        if remote in queued_nodes:
            return False
        queued_nodes.add(remote)
        await unbonded_send.send(remote)
        return True

    async def attempt_bond():
        while True:
            unbonded_node = await unbonded_recv.receive()
            if await protocol.bond(unbonded_node):
                logger.info(f'FOUND NODE {unbonded_node.uri()}')
                bonded_nodes.add(unbonded_node)

    async def attempt_random_lookup():
        while True:
            while len(bonded_nodes) == 0:
                await trio.sleep(1)
            bonded_node = bonded_nodes.pop()  # only send one query to each remote

            target = random_nodeid()
            protocol._send_find_node(bonded_node, target)
            candidates = await protocol.wait_neighbours(bonded_node)
            results = [await enqueue_node(remote) for remote in candidates]
            logger.debug(
                'finished %s. %s of %s nodes were enqueued',
                bonded_node, sum(results), len(candidates)
            )

    async with trio.open_nursery() as nursery:
        for i in range(64):
            nursery.start_soon(attempt_bond)

        for i in range(2):
            nursery.start_soon(attempt_random_lookup)

        for node in bootnodes:
            await enqueue_node(node)


async def run(port, privkey, address, bootnodes) -> None:
    socket = trio.socket.socket(family=trio.socket.AF_INET, type=trio.socket.SOCK_DGRAM)
    await socket.bind(('0.0.0.0', port))
    logger.debug(f'successfully bound port={port}')

    # This is not what I want, because I also want to run some of my own tasks
    service = DiscoveryProtocol(privkey, address, socket)

    async with trio.open_nursery() as nursery:
        manager = TrioManager(service)
        nursery.start_soon(manager.run)
        await manager.wait_started()

        await crawl(service, bootnodes)

        nursery.cancel_scope.cancel()

    return


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('-port', type=int, default=30304)
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(name)s: %(message)s')
    logging.getLogger('async_service.Manager').setLevel(logging.INFO)

    logger.debug(f'using port: {args.port}')
    address = kademlia.Address('0.0.0.0', args.port, args.port)

    privkey = ecies.generate_privkey()

    logger.debug(f'using mainnet bootnodes')
    bootnodes = [kademlia.Node.from_uri(enode) for enode in constants.MAINNET_BOOTNODES]

    try:
        trio.run(run, args.port, privkey, address, bootnodes)
    except KeyboardInterrupt:
        logger.info('shut down in response to Ctrl-C')


if __name__ == "__main__":
    main()
