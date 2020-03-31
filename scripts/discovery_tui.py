#!/usr/bin/env python

import argparse
import logging
import dataclasses

import os
import sys
import termios
import trio
import tty
import signal
import io
import time

# This script uses prompt_toolkit rather than input because of an annoying bug involving
# readline and Ctrl-C. Python intercepts the Ctrl-C and raises it in all threads as a
# KeyboardInterrupt. That messes with readline_until_enter_or_signal, which does some
# cleanup if a SIGINT comes in during the call to rl_callback_read_char(). Python never
# gives backgrounds signals, so that cleanup never happens, so Ctrl-C causes readline to
# leave the terminal in a bad state. prompt_toolkit does not suffer from this problem.
from prompt_toolkit import prompt

from eth_utils import int_to_big_endian, big_endian_to_int

from p2p import constants, ecies, kademlia

import discv4


logger = logging.getLogger('discovery_tui')


async def accept_input(nursery, state):
    while True:
        try:
            # This is dangerous, "cancellable" means we abandon the background thread and
            # continue onward. This is only safe because we immediately quit the program.
            # without cancellable=True Ctrl-C would not interrupt this task until it
            # resumed (because more input was read)
            command = await trio.to_thread.run_sync(prompt, '> ', cancellable=True)
        except EOFError:
            # the user probably pressed Ctrl-D
            nursery.cancel_scope.cancel()
            return
        await run_command(command.strip(), nursery, state)


@dataclasses.dataclass
class State:
    port: int
    server: None


async def run_command(command, nursery, state):
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")

    bootnodes_parser = subparsers.add_parser('bootnodes')

    listen_parser = subparsers.add_parser('listen')
    listen_parser.add_argument('port', type=int, default=state.port, nargs='?')

    ping_parser = subparsers.add_parser('ping')
    ping_parser.add_argument('server', type=int)

    random_nodeid_parser = subparsers.add_parser('random_nodeid')

    send_find_node_parser = subparsers.add_parser('send_find_node')
    send_find_node_parser.add_argument('server', type=int)
    send_find_node_parser.add_argument('target', type=int)

    args = parser.parse_args(command.split(' '))

    bootnodes = [kademlia.Node.from_uri(enode) for enode in constants.MAINNET_BOOTNODES]

    if args.command == 'bootnodes':
        for i, bootnode in enumerate(bootnodes):
            logger.info(f'[{i:02}] {bootnode}')
    elif args.command == 'random_nodeid':
        randomid = discv4.random_nodeid()
        logger.info(f'random id: {randomid}')
    elif args.command == 'listen':
        if state.server != None:
            logger.error('server already started')
            return

        logger.info(f'starting to listen on port {args.port}')
        logger.debug(f'using mainnet bootnodes')
        address = kademlia.Address('0.0.0.0', args.port, args.port)
        privkey = ecies.generate_privkey()

        socket = trio.socket.socket(family=trio.socket.AF_INET, type=trio.socket.SOCK_DGRAM)
        await socket.bind(('0.0.0.0', args.port))
        state.port = args.port

        server = discv4.Server(nursery, socket, privkey, address)
        nursery.start_soon(server.listen)

        state.server = server
    elif args.command == 'ping':
        if state.server == None:
            logger.error('must be listening to send packets')
            return

        node = bootnodes[args.server]
        await state.server.ping(node)
    elif args.command == 'send_find_node':
        if state.server == None:
            logger.error('must be listening to send packets')
            return

        node = bootnodes[args.server]
        targetid = args.target
        await state.server.proto.send_find_node(node, targetid)
    else:
        logger.error(f'Unknown command: {args.command}')


class PromptFriendlyConsoleHandler(logging.Handler):
    """
    Does a little dance such that messages emitted to the console do not mess with the
    active prompt. It will appear as though the prompt is glued to the bottom of the
    screen, as the lines about it 
    """
    def emit(self, record) -> None:
        try:
            msg = self.format(record)

            sys.stdout.write('\x1b\x37'); # save cursor position
            sys.stdout.write('\n')  # scroll everything up one line
            sys.stdout.write('\x9bA'); # move the cursor up one line
            sys.stdout.write('\x9bL'); # insert a line here
            sys.stdout.write(msg)
            sys.stdout.write('\x1b\x38'); # restore the cursor position
            sys.stdout.write('\x9bB'); # move the cursor back down a line
            sys.stdout.flush()
        except Exception:  # the catch-all is apparently appropriate here
            self.handleError(record)


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-port', type=int, default=30303)
    args = parser.parse_args()

    handler = PromptFriendlyConsoleHandler()
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s %(levelname)s: %(message)s',
        handlers=[handler],
    )

    state = State(port=args.port, server=None)

    try:
        async with trio.open_nursery() as nursery:
            nursery.start_soon(accept_input, nursery, state)
    except KeyboardInterrupt:
        print('Ctrl-C')


if __name__ == '__main__':
    trio.run(main)
