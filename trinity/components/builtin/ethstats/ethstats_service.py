import asyncio
import platform

import websockets

from lahja import EndpointAPI

from eth.abc import ChainAPI

from p2p.service import (
    BaseService,
)

from trinity.config import (
    Eth1AppConfig,
)
from trinity.constants import (
    SYNC_LIGHT,
    TO_NETWORKING_BROADCAST_CONFIG,
)
from trinity.chains.light_eventbus import (
    EventBusLightPeerChain,
)
from trinity.db.eth1.header import AsyncHeaderDB
from trinity.db.manager import DBClient
from trinity._utils.version import (
    construct_trinity_client_identifier,
)

from trinity.extensibility.component import (
    TrinityBootInfo,
)
from trinity.components.builtin.ethstats.ethstats_client import (
    EthstatsClient,
    EthstatsMessage,
    EthstatsData,
    timestamp_ms,
)
from trinity.protocol.common.events import (
    PeerCountRequest,
)


class EthstatsService(BaseService):
    def __init__(
        self,
        boot_info: TrinityBootInfo,
        event_bus: EndpointAPI,
        server_url: str,
        server_secret: str,
        node_id: str,
        node_contact: str,
        stats_interval: int,
    ) -> None:
        super().__init__()

        self.boot_info = boot_info
        self.event_bus = event_bus

        self.server_url = server_url
        self.server_secret = server_secret
        self.node_id = node_id
        self.node_contact = node_contact
        self.stats_interval = stats_interval

        self.chain = self.get_chain()

    async def _run(self) -> None:
        while self.is_operational:
            self.logger.info('Connecting to %s...', self.server_url)
            async with websockets.connect(self.server_url) as websocket:
                client: EthstatsClient = EthstatsClient(
                    websocket,
                    self.node_id,
                    token=self.cancel_token,
                )

                self.run_daemon_task(self.server_handler(client))
                self.run_daemon_task(self.statistics_handler(client))

                await client.run()
                if self.is_operational and not client.is_operational:
                    self.logger.info('Connection to %s closed', self.server_url)
                    self.logger.info('Reconnecting in 5s...')
                    await self.sleep(5)

    # Wait for messages from server, respond when they arrive
    async def server_handler(self, client: EthstatsClient) -> None:
        while self.is_operational:
            message: EthstatsMessage = await client.recv()

            if message.command == 'node-pong':
                await client.send_latency((timestamp_ms() - message.data['clientTime']) // 2)
            elif message.command == 'history':
                # TODO: send actual history
                pass
            else:
                self.logger.info('Server message received')

    # Periodically send statistics and ping server to calculate latency
    async def statistics_handler(self, client: EthstatsClient) -> None:
        await client.send_hello(self.server_secret, self.get_node_info())

        while self.is_operational:
            await client.send_node_ping()
            await client.send_stats(await self.get_node_stats())
            await client.send_block(self.get_node_block())

            await self.sleep(self.stats_interval)

    def get_node_info(self) -> EthstatsData:
        """Getter for data that should be sent once, on start-up."""
        return {
            'name': self.node_id,
            'contact': self.node_contact,
            'node': construct_trinity_client_identifier(),
            'net': self.boot_info.trinity_config.network_id,
            'port': self.boot_info.trinity_config.port,
            'os': platform.system(),
            'os_v': platform.release(),
            'client': '0.1.1',
            'canUpdateHistory': False,
        }

    def get_node_block(self) -> EthstatsData:
        """Getter for data that should be sent on every new chain tip change."""
        head = self.chain.get_canonical_head()

        return {
            'number': head.block_number,
            'hash': head.hex_hash,
            'difficulty': head.difficulty,
            'totalDifficulty': self.chain.get_score(head.hash),
            'transactions': [],
            'uncles': [],
        }

    async def get_node_stats(self) -> EthstatsData:
        """Getter for data that should be sent periodically."""
        try:
            peer_count = (await self.wait(
                self.event_bus.request(
                    PeerCountRequest(),
                    TO_NETWORKING_BROADCAST_CONFIG,
                ),
                timeout=1
            )).peer_count
        except asyncio.TimeoutError:
            self.logger.warning("Timeout: PeerPool did not answer PeerCountRequest")
            peer_count = 0

        return {
            'active': True,
            'uptime': 100,
            'peers': peer_count,
        }

    def get_chain(self) -> ChainAPI:
        app_config = self.boot_info.trinity_config.get_app_config(Eth1AppConfig)
        chain_config = app_config.get_chain_config()

        chain: ChainAPI
        base_db = DBClient.connect(self.boot_info.trinity_config.database_ipc_path)

        if self.boot_info.args.sync_mode == SYNC_LIGHT:
            header_db = AsyncHeaderDB(base_db)
            chain = chain_config.light_chain_class(
                header_db,
                peer_chain=EventBusLightPeerChain(self.event_bus)
            )
        else:
            chain = chain_config.full_chain_class(base_db)

        return chain
