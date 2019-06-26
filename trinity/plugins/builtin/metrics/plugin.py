import asyncio
from argparse import (
    ArgumentParser,
    Namespace,
    _SubParsersAction,
)
from collections import defaultdict
import os
from pathlib import Path
import socket
import time
import pickle
from typing import Dict

import urwid

from trinity.config import (
    Eth1AppConfig,
    TrinityConfig,
)

from trinity.endpoint import (
    TrinityEventBusEndpoint,
)
from trinity.extensibility import (
    BaseMainProcessPlugin,
    AsyncioIsolatedPlugin,
)
from trinity._utils.shutdown import (
    exit_with_endpoint_and_services,
)

from p2p.service import (
    BaseService,  # does this belong in p2p.*?
)

from trinity.metrics import Metric

from lahja import BaseEvent

from .tui import UrwidInterface


IPC_FILE_NAME = "metrics.ipc"


def ipc_path(config: TrinityConfig) -> Path:
    return config.ipc_dir / IPC_FILE_NAME


class MetricsServer(BaseService):
    def __init__(self, ipc_path: Path, event_bus) -> None:
        super().__init__()

        self.ipc_path = ipc_path
        self.event_bus = event_bus
        self.server = None

        self.metrics: Dict[str, int] = dict()

    async def _run(self) -> None:
        self.server = await asyncio.start_unix_server(
            self.handle_connection,
            str(self.ipc_path),
            loop=self.get_event_loop(),
        )
        self.logger.info('Metrics server started: %s', self.ipc_path)

        async for event in self.wait_iter(self.event_bus.stream(Metric)):
            # self.logger.debug(f'received event: {event.metric}, {event.value}')
            self.metrics[event.metric] = event.value

        await self.cancel_token.wait()

    async def handle_connection(self, reader: asyncio.StreamReader,
                                      writer: asyncio.StreamWriter) -> None:
        while not self.is_cancelled:
            await self.wait(reader.readline())

            if reader.at_eof():
                # the remote no longer exists
                return

            pickled = pickle.dumps(self.metrics)
            size = len(pickled)
            writer.write(
                size.to_bytes(4, "little") + pickled
            )

        writer.close()

    async def _cleanup(self) -> None:
        self.logger.info('Metrics server closing')
        self.server.close()
        await self.server.wait_closed()
        self.ipc_path.unlink()


class MetricsServerPlugin(AsyncioIsolatedPlugin):
    """
    Collects metrics and makes them available to the metrics plugin
    """

    @property
    def name(self) -> str:
        return "Metrics Server"

    def on_ready(self, manager_eventbus: TrinityEventBusEndpoint) -> None:
        self.start()

    @classmethod
    def configure_parser(cls, arg_parser: ArgumentParser, subparser) -> None:
        pass

    def do_start(self) -> None:
        config = self.boot_info.trinity_config
        server = MetricsServer(ipc_path(config), self.event_bus)

        asyncio.ensure_future(exit_with_endpoint_and_services(self.event_bus, server))
        asyncio.ensure_future(server.run())


async def read_stats(reader: asyncio.StreamReader) -> Dict:
    raw_size = await reader.readexactly(4)
    size = int.from_bytes(raw_size, "little")
    pickled = await reader.readexactly(size)
    return pickle.loads(pickled)


class MetricsPlugin(BaseMainProcessPlugin):
    """
    Continously try to map external to internal ip address/port using the
    Universal Plug 'n' Play (upnp) standard.
    """

    @property
    def name(self) -> str:
        return "Metrics"

    @classmethod
    def configure_parser(cls,
                         arg_parser: ArgumentParser,
                         subparser: _SubParsersAction) -> None:

        metrics_parser = subparser.add_parser(
            "metrics",
            help='Open a simple metrics interface'
        )

        metrics_parser.set_defaults(func=cls.run_console)

    @classmethod
    async def communicate(cls, path: str, interface: UrwidInterface) -> None:
        interface.set_footer_text('hello')

        try:
            reader, writer = await asyncio.open_unix_connection(path)
            interface.set_footer_text("connected")
        except:
            interface.stop()
            raise

        try:
            while True:
                writer.write(b'\n')
                await writer.drain()

                stats = await read_stats(reader)
                interface.set_data(stats)

                await asyncio.sleep(1)

        except asyncio.IncompleteReadError:
            interface.stop()

    @classmethod
    def run_console(cls, args: Namespace, trinity_config: TrinityConfig) -> None:
        interface = UrwidInterface()

        path = str(ipc_path(trinity_config))
        loop = asyncio.get_event_loop()
        # asyncio.ensure_future(cls.communicate(path, interface))

        try:
            loop.run_until_complete(interface.run())
        except KeyboardInterrupt:
            pass
        finally:
            interface.stop()  # cleanup the screen

    @classmethod
    def brun_console(cls, args: Namespace, trinity_config: TrinityConfig) -> None:
        #cols, lines = os.get_terminal_size()
        #print(cols, lines)
        #return

        with socket.socket(family=socket.AF_UNIX) as sock:
            sock.connect(str(ipc_path(trinity_config)))
            read = sock.makefile(mode='rb')
            write = sock.makefile(mode='w')

            try:
                while True:
                    write.write('\n')
                    write.flush()

                    raw_size = read.read(4)
                    size = int.from_bytes(raw_size, "little")
                    pickled = read.read(size)
                    stats = pickle.loads(pickled)

                    print(f'stats: {stats}')
                    time.sleep(1)
            except BrokenPipeError:
                print('remote disconnected')
