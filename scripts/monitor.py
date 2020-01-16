#!/usr/bin/env python

import argparse
import asyncio
import asyncio.streams
import contextlib
import datetime
import logging
import os
import struct
import snappy
import sys
import plyvel
import traceback

from sqlalchemy import (
    create_engine, MetaData, Table, Column, Integer, Text, DateTime,
)
from sqlalchemy.schema import CreateTable, DropTable

from cancel_token import CancelToken, OperationCancelled

import rlp
import rlp.exceptions
from rlp.sedes import CountableList

from eth_utils import DEBUG2_LEVEL_NUM, humanize_hash

from p2p import kademlia, ecies
from p2p.auth import HandshakeInitiator, handshake
from p2p.handshake import dial_out, DevP2PHandshakeParams
from p2p.constants import DEVP2P_V5
from p2p.exceptions import (
    HandshakeFailure,
    HandshakeFailureTooManyPeers,
    MalformedMessage,
    NoMatchingPeerCapabilities,
    UnreachablePeer,
    PeerConnectionLost,
)
from p2p.service import run_service
from p2p.p2p_proto import P2PProtocolV5, Disconnect, Ping
from p2p.p2p_api import P2PAPI

from trinity.protocol.eth.commands import (
    Transactions,
    NewBlockHashes, NewBlock,
    GetBlockHeaders, BlockHeaders,
    GetNodeData, NodeData,
    GetBlockBodies, BlockBodies,
)

from eth.constants import GENESIS_DIFFICULTY
from eth.chains.mainnet import MAINNET_GENESIS_HEADER

from trinity.exceptions import WrongNetworkFailure, WrongGenesisFailure

from trinity.protocol.eth.proto import ETHProtocol
from trinity.protocol.eth.payloads import StatusPayload
from trinity.protocol.eth.handshaker import ETHHandshaker

from eth.rlp.transactions import BaseTransactionFields
from eth.rlp.headers import BlockHeader


logger = logging.getLogger(__file__)


# Utilities for reading from the Geth database


class GethKeys:
    # from https://github.com/ethereum/go-ethereum/blob/master/core/rawdb/schema.go
    DatabaseVersion = b'DatabaseVersion'
    HeadBlock = b'LastBlock'

    headerPrefix = b'h'
    headerNumberPrefix = b'H'
    headerHashSuffix = b'n'

    blockBodyPrefix = b'b'
    blockReceiptsPrefix = b'r'

    @classmethod
    def header_hash_for_block_number(cls, block_number: int) -> bytes:
        "The key to get the hash of the header with the given block number"
        packed_block_number = struct.pack('>Q', block_number)
        return cls.headerPrefix + packed_block_number + cls.headerHashSuffix

    @classmethod
    def block_number_for_header_hash(cls, header_hash: bytes) -> bytes:
        "The key to get the block number of the header with the given hash"
        return cls.headerNumberPrefix + header_hash

    @classmethod
    def block_header(cls, block_number: int, header_hash: bytes) -> bytes:
        packed_block_number = struct.pack('>Q', block_number)
        return cls.headerPrefix + packed_block_number + header_hash

    @classmethod
    def block_body(cls, block_number: int, header_hash: bytes) -> bytes:
        packed_block_number = struct.pack('>Q', block_number)
        return cls.blockBodyPrefix + packed_block_number + header_hash

    @classmethod
    def block_receipts(cls, block_number: int, header_hash: bytes) -> bytes:
        packed_block_number = struct.pack('>Q', block_number)
        return cls.blockReceiptsPrefix + packed_block_number + header_hash


class GethFreezerIndexEntry:
    def __init__(self, filenum: int, offset: int):
        self.filenum = filenum
        self.offset = offset

    @classmethod
    def from_bytes(cls, data: bytes) -> 'GethFreezerIndexEntry':
        assert len(data) == 6
        filenum, offset = struct.unpack('>HI', data)
        return cls(filenum, offset)

    def __repr__(self):
        return f'IndexEntry(filenum={self.filenum}, offset={self.offset})'


class GethFreezerTable:
    def __init__(self, ancient_path, name, uses_compression):
        self.ancient_path = ancient_path
        self.name = name
        self.uses_compression = uses_compression

        self.index_file = open(os.path.join(ancient_path, self.index_file_name), 'rb')
        stat_result = os.stat(self.index_file.fileno())
        index_file_size = stat_result.st_size
        assert index_file_size % 6 == 0, index_file_size
        self.entries = index_file_size // 6

        self._data_files = dict()

    @property
    def index_file_name(self):
        suffix = 'cidx' if self.uses_compression else 'ridx'
        return f'{self.name}.{suffix}'

    def data_file_name(self, number: int):
        suffix = 'cdat' if self.uses_compression else 'rdat'
        return f'{self.name}.{number:04d}.{suffix}'

    def _data_file(self, number: int):
        if number not in self._data_files:
            path = os.path.join(self.ancient_path, self.data_file_name(number))
            data_file = open(path, 'rb')
            self._data_files[number] = data_file

        return self._data_files[number]

    def get(self, number: int) -> bytes:
        assert number < self.entries

        self.index_file.seek(number * 6)
        entry_bytes = self.index_file.read(6)
        start_entry = GethFreezerIndexEntry.from_bytes(entry_bytes)

        # What happens if we're trying to read the last item? Won't this fail?
        # Is there always one extra entry in the index file?
        self.index_file.seek((number + 1) * 6)
        entry_bytes = self.index_file.read(6)
        end_entry = GethFreezerIndexEntry.from_bytes(entry_bytes)

        if start_entry.filenum != end_entry.filenum:
            # Duplicates logic from freezer_table.go:getBounds
            start_entry = GethFreezerIndexEntry(end_entry.filenum, offset=0)

        data_file = self._data_file(start_entry.filenum)
        data_file.seek(start_entry.offset)
        data = data_file.read(end_entry.offset - start_entry.offset)

        if not self.uses_compression:
            return data

        return snappy.decompress(data)

    def __del__(self) -> None:
        for f in self._data_files.values():
            f.close()
        self.index_file.close()

    @property
    def last_index(self):
        self.index_file.seek(-6, 2)
        last_index_bytes = self.index_file.read(6)
        return GethFreezerIndexEntry.from_bytes(last_index_bytes)

    @property
    def first_index(self):
        self.index_file.seek(0)
        first_index_bytes = self.index_file.read(6)
        return GethFreezerIndexEntry.from_bytes(first_index_bytes)


class BlockBody(rlp.Serializable):
    "This is how geth stores block bodies"
    fields = [
        ('transactions', CountableList(BaseTransactionFields)),
        ('uncles', CountableList(BlockHeader)),
    ]

    def __repr__(self) -> str:
        return f'BlockBody(txns={self.transactions}, uncles={self.uncles})'


class GethDatabase:
    def __init__(self, path):
        self.db = plyvel.DB(
            path,
            create_if_missing=False,
            error_if_exists=False,
            max_open_files=16
        )

        ancient_path = os.path.join(path, 'ancient')
        self.ancient_hashes = GethFreezerTable(ancient_path, 'hashes', False)
        self.ancient_headers = GethFreezerTable(ancient_path, 'headers', True)
        self.ancient_bodies = GethFreezerTable(ancient_path, 'bodies', True)
        self.ancient_receipts = GethFreezerTable(ancient_path, 'receipts', True)

        if self.database_version != b'\x07':
            raise Exception(f'geth database version {self.database_version} is not supported')

    @property
    def database_version(self) -> bytes:
        raw_version = self.db.get(GethKeys.DatabaseVersion)
        return rlp.decode(raw_version)

    @property
    def last_block_hash(self) -> bytes:
        return self.db.get(GethKeys.HeadBlock)

    def block_num_for_hash(self, header_hash: bytes) -> int:
        raw_num = self.db.get(GethKeys.block_number_for_header_hash(header_hash))
        return struct.unpack('>Q', raw_num)[0]

    def block_header(self, block_number: int, header_hash: bytes = None) -> BlockHeader:
        if header_hash is None:
            header_hash = self.header_hash_for_block_number(block_number)

        raw_data = self.db.get(GethKeys.block_header(block_number, header_hash))
        if raw_data is not None:
            return rlp.decode(raw_data, sedes=BlockHeader)

        raw_data = self.ancient_headers.get(block_number)
        return rlp.decode(raw_data, sedes=BlockHeader)

    def header_hash_for_block_number(self, block_number: int) -> bytes:
        # This needs to check the ancient db (freezerHashTable)
        result = self.db.get(GethKeys.header_hash_for_block_number(block_number))

        if result is not None:
            return result

        return self.ancient_hashes.get(block_number)

    def block_body(self, block_number: int, header_hash: bytes = None):
        if header_hash is None:
            header_hash = self.header_hash_for_block_number(block_number)

        raw_data = self.db.get(GethKeys.block_body(block_number, header_hash))
        if raw_data is not None:
            return rlp.decode(raw_data, sedes=BlockBody)

        raw_data = self.ancient_bodies.get(block_number)
        return rlp.decode(raw_data, sedes=BlockBody)

    def block_receipts(self, block_number: int, header_hash: bytes = None):
        if header_hash is None:
            header_hash = self.header_hash_for_block_number(block_number)

        raw_data = self.db.get(GethKeys.block_receipts(block_number, header_hash))
        if raw_data is not None:
            return raw_data

        raw_data = self.ancient_receipts.get(block_number)
        return raw_data


# Utilities for reading / writing a nodedb


metadata = MetaData()
nodes = Table(
    'nodes', metadata,
    Column('enode', Text, primary_key=True),
)

blacklist = Table(
    'blacklist', metadata,
    Column('enode', Text, primary_key=True),
    Column('reason', Text),
    Column('add_time', DateTime),
)

deferred_nodes = Table(
    'deferred', metadata,
    Column('enode', Text, primary_key=True),
    Column('reason', Text),
    Column('expire_time', DateTime),
)

schema_version = Table(
    'schema_version', metadata,
    Column('version', Integer),
)

current_schema = 2


def does_table_exist(conn, table_name) -> bool:
    return bool(conn.scalar(
        f"SELECT name FROM sqlite_master WHERE type='table' AND name='{nodes.name}';"
    ))


def upgrade_schema(conn) -> None:
    if does_table_exist(conn, 'schema_version'):
        reported_schema = conn.scalar('SELECT version FROM schema_version;')
        if reported_schema == current_schema:
            return

    conn.execute('DROP TABLE IF EXISTS nodes;')
    conn.execute(CreateTable(nodes))

    conn.execute('DROP TABLE IF EXISTS blacklist;')
    conn.execute(CreateTable(blacklist))

    conn.execute('DROP TABLE IF EXISTS deferred;')
    conn.execute(CreateTable(deferred_nodes))

    conn.execute('DROP TABLE IF EXISTS schema_version;')
    conn.execute(CreateTable(schema_version))
    conn.execute(f'INSERT INTO schema_version (version) VALUES ({current_schema});')


def blacklist_node(conn, remote, reason):
    conn.execute(blacklist.insert().values(
        enode=remote.uri(),
        reason=reason,
        add_time=datetime.datetime.now(),
    ))


def defer_node(conn, remote, reason, delay):
    conn.execute(deferred_nodes.insert().values(
        enode=remote.uri(),
        reason=reason,
        expire_time=datetime.datetime.now() + delay,
    ))


def should_connect(conn, node):
    enode = node.uri()

    res = conn.execute(blacklist.select(blacklist.c.enode == enode))
    f = res.first()
    if f is not None:
        return False

    res = conn.execute(deferred_nodes.select(deferred_nodes.c.enode == enode))
    f = res.first()
    if f is not None:
        try:
            expire_time = f.expire_time
        except AttributeError:
            logger.error(f'row has no expiration: {f} enode={enode}')
            return False

        if expire_time > datetime.datetime.now():
            return False

        #logger.debug(f'{node} was deferred but trying again. expire_time={f.expire_time} reason={f.reason}')
        conn.execute(
            deferred_nodes.delete(deferred_nodes.c.enode == enode)
        )

    return True


class PeerCountTracker:
    def __init__(self):
        self.peer_count = 0
        self.peer_disconnected_event = asyncio.Event()

    @contextlib.contextmanager
    def track(self, remote) -> None:
        self.peer_count += 1
        logger.info(f'connected. remote={remote} peer_count={self.peer_count}')
        try:
            yield
        finally:
            self.peer_count -= 1
            self.peer_disconnected_event.set()
            logger.info(f'disconnected. remote={remote} peer_count={self.peer_count}')

    async def wait_until_allowed_to_connect(self):
        while self.peer_count >= 80:
            await self.peer_disconnected_event.wait()
            if not self.peer_disconnected_event.is_set():
                # another coro woke up before this one
                continue
            # this coro woke first, signal to the others that they're too late
            self.peer_disconnected_event.clear()
            break


class ConnectedPeerTracker:
    "TODO: merge this in with PeerCountTracker"

    def __init__(self):
        self.peers = set()

    def can_connect(self, remote):
        return not remote in self.peers

    def connect(self, remote):
        self.peers.add(remote)

    def disconnect(self, remote):
        self.peers.remove(remote)

    @contextlib.contextmanager
    def lock(self, remote):
        self.connect(remote)
        try:
            yield
        finally:
            self.disconnect(remote)


async def feed_queue(candidate_enode_queue, enode_file_name):
    """
    More efficient then cycling through every node, looking for deferred ones which are
    ready to be tried again, would be to maintain some kind of priority queue with the
    deferrals which will expire first up front. Alternatively, this could query the
    database for the next node to attempt to connect to. That would work well with a
    future requirement that some other process can crawl the DHT and add candidates, it
    could add them to the database.
    """
    while True:
        with open(enode_file_name) as f:
            for enode_line in f:
                enode = enode_line.strip()
                node = kademlia.Node.from_uri(enode)
                await candidate_enode_queue.put(node)
        logger.info('ran out of candidates, running through the list again')


async def connect_from_queue(engine, candidate_enode_queue, peer_tracker, connect_tracker, gethdb):
    tasks = []

    conn = engine.connect()
    try:
        while True:
            # if we connect to too many remotes then the event loop gets bogged down
            await peer_tracker.wait_until_allowed_to_connect()

            node = await candidate_enode_queue.get()

            if should_connect(conn, node) and connect_tracker.can_connect(node.uri()):
                with connect_tracker.lock(node.uri()):
                    connection = await connect(conn, node)
                if connection:
                    connect_tracker.connect(node.uri())
                    task = asyncio.ensure_future(
                        process_with_tracking(engine, connection, peer_tracker, connect_tracker, gethdb)
                    )
                    tasks.append((connection, task))
    except asyncio.CancelledError:
        logger.info('connect_from_queue: cancelled')
        raise
    except BaseException:
        logger.exception('exception during connect_from_queue')
        raise
    finally:
        logger.info('connect_from_queue: cleaning up all tasks')
        conn.close()
        for (connection, task) in tasks:
            if not connection.is_cancelled:
                await connection.cancel()
            task.cancel()
        for (_connection, task) in tasks:
            await task


async def connect_all(enode_file_name, gethdb):
    engine = create_engine('sqlite:///nodedb.sqlite')

    try:
        queue = asyncio.Queue(maxsize=10)
        conn = engine.connect()
        upgrade_schema(conn)
        conn.close()

        tracker = PeerCountTracker()
        connect_tracker = ConnectedPeerTracker()

        feeder = asyncio.ensure_future(feed_queue(queue, enode_file_name))
        readers = []
        for _ in range(20):
            readers.append(asyncio.ensure_future(
                connect_from_queue(engine, queue, tracker, connect_tracker, gethdb)
            ))

        try:
            await asyncio.gather(*readers)
            logger.debug('gather finished')
        except BaseException:
            logger.exception('exception during gather')
            raise
        finally:
            logger.info('cancelling all connect threads')
            feeder.cancel()
            for reader in readers:
                reader.cancel()
            try:
                await feeder
            except asyncio.CancelledError:
                pass
            for reader in readers:
                try:
                    await reader
                except asyncio.CancelledError:
                    pass
    except BaseException:
        logger.exception('exception during connect_all')
        raise


async def connect(conn, remote):
    privkey = ecies.generate_privkey()

    p2p_handshake_params = DevP2PHandshakeParams(
        client_version_string="TrinityAnalytics/v0.0.1/linux-amd64/python3.7.5",
        listen_port=30306,
        version=DEVP2P_V5,
    )

    eth_handshake_params = StatusPayload(
        head_hash=MAINNET_GENESIS_HEADER.hash,
        total_difficulty=GENESIS_DIFFICULTY,
        genesis_hash=MAINNET_GENESIS_HEADER.hash,
        network_id=1,
        version=ETHProtocol.version,
    )

    protocol_handshakers = (
        ETHHandshaker(eth_handshake_params),
    )

    try:
        token = CancelToken("initiator")
        connection = await token.cancellable_wait(
            dial_out(remote, privkey, p2p_handshake_params, protocol_handshakers, token),
            timeout=5
        )
    except asyncio.TimeoutError:
        defer_node(conn, remote, 'TimeoutError', datetime.timedelta(hours=1))
        logger.error(f'connecting timed out remote={remote}')
        return
    except WrongNetworkFailure as e:
        blacklist_node(conn, remote, 'WrongNetworkFailure')
        logger.error(f'WrongNetworkFailure={e} remote={remote}')
        return
    except HandshakeFailureTooManyPeers:
        defer_node(conn, remote, 'TooManyPeers', datetime.timedelta(minutes=5))
        logger.error(f'too many peers remote={remote}')
        return
    except NoMatchingPeerCapabilities as e:
        blacklist_node(conn, remote, 'NoMatchingPeerCapabilities')
        logger.error(f'no matching caps: %s remote={remote}', e.remote_capabilities)
        return
    except UnreachablePeer:
        defer_node(conn, remote, 'UnreachablePeer', datetime.timedelta(days=1))
        logger.error(f'unreachable peer remote={remote}')
        return
    except WrongGenesisFailure as e:
        blacklist_node(conn, remote, 'WrongGenesisFailure')
        logger.error(f'WrongGenesis: {e} remote={remote}')
        return
    except HandshakeFailure as e:
        defer_node(conn, remote, 'UnreachablePeer', datetime.timedelta(hours=2))
        logger.error(f'handshake failure: {e} remote={remote}')
        return
    except PeerConnectionLost:
        defer_node(conn, remote, 'PeerConnectionLost', datetime.timedelta(hours=1))
        logger.error(f'peer connection lost remote={remote}')
        return
    except MalformedMessage:
        defer_node(conn, remote, 'MalformedMessage', datetime.timedelta(hours=1))
        logger.error(f'MalformedMessage remote={remote}')
        return
    except ConnectionResetError:
        defer_node(conn, remote, 'ConnectionResetError', datetime.timedelta(hours=1))
        logger.error(f'ConnectionResetError remote={remote}')
        return
    except asyncio.CancelledError:
        logger.debug(f'connect was cancelled remote={remote}')
        raise
    except rlp.exceptions.DeserializationError:
        defer_node(conn, remote, 'DeserializationError', datetime.timedelta(hours=2))
        logger.error(f'DeserializationError remote={remote}')
        return
    except Exception as e:
        # TODO: this should trigger a full shutdown
        logger.exception(f'exception during connect remote={remote}')
        raise

    logger.info(f'established connection: {connection.session.remote} {connection.safe_client_version_string}')
    return connection


async def process_with_tracking(engine, connection, peer_tracker, connect_tracker, gethdb):
    try:
        with peer_tracker.track(connection.session.remote):
            await process(engine, connection, gethdb)
    except BaseException:
        logger.exception('exception during process_with_tracking')
        raise
    finally:
        enode = connection.session.remote.uri()
        connect_tracker.disconnect(enode)


async def process(engine, connection, gethdb):
    eth_proto = connection.get_protocol_by_type(ETHProtocol)
    max_header = gethdb.block_num_for_hash(gethdb.last_block_hash)

    def kick_node():
        logger.info(f'remote={connection.session.remote} error=NeverAnnouncedBlock')
        blacklist_node(engine, connection.session.remote, 'NeverAnnouncedABlock')
        asyncio.create_task(connection.cancel())

    # If the node doesn't send us any block announcements within 2 minutes it probably
    # never will, disconnect and try to find a different node.
    loop = asyncio.get_event_loop()
    timeout = loop.call_later(delay=60, callback=kick_node)

    async def handle_eth_message(connection, cmd):
        if isinstance(cmd, GetBlockHeaders):
            if gethdb is None:
                logger.info(
                    f'{connection.session.remote} '
                    f'GetBlockHeaders '
                    f'payload={cmd.payload} '
                    f'error=no-gethdb'
                )
                await asyncio.sleep(1)  # a short delay so syncing peers like us less
                eth_proto.send(BlockHeaders(tuple()))
                return

            if gethdb is None or cmd.payload.max_headers != 1:
                # answer the checkpoint queries, don't help other peers sync
                logger.info(
                    f'{connection.session.remote} '
                    f'GetBlockHeaders '
                    f'payload={cmd.payload} '
                    f'error=request-too-big'
                )
                await asyncio.sleep(1)  # a short delay so syncing peers like us less
                eth_proto.send(BlockHeaders(tuple()))
                return

            if not isinstance(cmd.payload.block_number_or_hash, int):
                logger.info(
                    f'{connection.session.remote} '
                    f'GetBlockHeaders '
                    f'payload={cmd.payload} '
                    f'error=requested-header-by-hash'
                )
                await asyncio.sleep(1)  # a short delay so syncing peers like us less
                eth_proto.send(BlockHeaders(tuple()))
                return

            if cmd.payload.block_number_or_hash > max_header:
                logger.info(
                    f'{connection.session.remote} '
                    f'GetBlockHeaders '
                    f'payload={cmd.payload} '
                    f'error=requested-header-too-recent'
                )
                await asyncio.sleep(1)  # a short delay so syncing peers like us less
                eth_proto.send(BlockHeaders(tuple()))
                return

            logger.info(
                f'{connection.session.remote} '
                f'GetBlockHeaders '
                f'payload={cmd.payload} '
            )
            # TODO: this read blocks the main thread. that's bad!
            header = gethdb.block_header(cmd.payload.block_number_or_hash)
            eth_proto.send(BlockHeaders([header]))
        elif isinstance(cmd, GetNodeData):
            logger.info(f'{connection.session.remote} GetNodeData (sending empty response)')
            await asyncio.sleep(1)  # a short delay so syncing peers like us less
            eth_proto.send(NodeData(tuple()))
        elif isinstance(cmd, GetBlockBodies):
            humanized = [block_hash.hex() for block_hash in cmd.payload]
            logger.info(f'{connection.session.remote} GetBlockBodies: {humanized} (sending empty response)')
            await asyncio.sleep(1)  # a short delay so syncing peers like us less
            eth_proto.send(BlockBodies(tuple()))
        elif isinstance(cmd, Transactions):
            pass
#            random_id = os.urandom(4).hex()  # for easier log correlation
#            logger.info(f'{connection.session.remote} Transactions: count={len(cmd.payload)} id={random_id}')
#            for txn in cmd.payload:
#                logger.info(
#                    f'transaction received: '
#                    f'remote={connection.session.remote} '
#                    f'len={len(rlp.encode(txn))} '
#                    f'hash={txn.hash.hex()} '
#                    f'id={random_id} '
#                )
#            logger.info(f'{connection.session.remote} Transactions: count={len(cmd.payload)}')

#            if len(cmd.payload) == 0:
#                logger.info(f'{connection.session.remote} empty Transactions')
        elif isinstance(cmd, NewBlockHashes):
            timeout.cancel()
            for new_block_hash in cmd.payload:
                logger.info(
                    f'NewBlockHashes: '
                    f'remote={connection.session.remote} '
                    f'hash={new_block_hash.hash.hex()} '
                    f'blocknum={new_block_hash.number} '
                )
                if new_block_hash.number > 9500000:
                    logger.info(f'remote={connection.session.remote} is ETC, kicking it')
                    blacklist_node(engine, connection.session.remote, 'BadBlockAnnouncement')
                    await connection.cancel()
                if new_block_hash.number < 8000000:
                    logger.info(f'remote={connection.session.remote} advertisement too low, kicking')
                    blacklist_node(engine, connection.session.remote, 'BadBlockAnnouncement')
                    await connection.cancel()
        elif isinstance(cmd, NewBlock):
            timeout.cancel()
            block = cmd.payload.block  # payload is protocol.eth.payloads.NewBlockPayload
            logger.info(
                f'NewBlock: '
                f'remote={connection.session.remote} '
                f'hash={block.header.hash.hex()} '
                f'blocknum={block.header.block_number} '
                f'size={len(rlp.encode(block))} '
                f'td={cmd.payload.total_difficulty} '
                f'transactions={len(block.transactions)} '
                f'uncles={len(block.uncles)}'
            )
            if block.header.block_number > 9500000:
                logger.info(f'remote={connection.session.remote} is ETC, kicking it')
                blacklist_node(engine, connection.session.remote, 'BadBlockAnnouncement')
                await connection.cancel()
            if block.header.block_number < 8000000:
                # There are ETH nodes, ETC nodes, and also nodes which seem to be doing
                # their own thing in he blocknum=6M range? No idea who they are but kick
                # them out so we can fit another ETH peer into the pool
                logger.info(f'remote={connection.session.remote} advertisement too low, kicking')
                blacklist_node(engine, connection.session.remote, 'BadBlockAnnouncement')
                await connection.cancel()
        else:
            logger.debug(f'unhandled ETH message {connection.session.remote}: {cmd}')

    async def handle_p2p_message(connection, cmd):
        if isinstance(cmd, Disconnect):
            logger.info(f'{connection.session.remote} disconnected reason={cmd.payload}')
        elif isinstance(cmd, Ping):
            logger.info(f'{connection.session.remote} ping, (sending pong)')
        else:
            logger.debug(f'unhandled p2p message {connection.session.remote}: {cmd}')

    # I wish that instead I could do some kind of for msg in connection.wait_iter():
    connection.add_protocol_handler(ETHProtocol, handle_eth_message)
    connection.add_protocol_handler(P2PProtocolV5, handle_p2p_message)
    async with run_service(connection):
        async with P2PAPI().apply(connection):
            connection.start_protocol_streams()
            try:
                await connection.cancellation()  # end processing if we disconnect
            except OperationCancelled:
                # cancelling the connection does not cancel the multiplexer...
                token = getattr(connection._multiplexer, '_multiplex_token', None)
                if token is not None:
                    token.trigger()

                # this try:except: probably belongs inside Service.cancellation(), if
                # we're waiting on cancellation then we probably don't want an exception
                # to tell us it's been cancelled!
                logger.debug('stopping because connection was cancelled')
            finally:
                timeout.cancel()

    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-enodes', type=str, required=True)
    parser.add_argument('-gethdb', type=str, required=False)
    args = parser.parse_args()

    # level=DEBUG2_LEVEL_NUM
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(name)s: %(message)s')
    logging.getLogger('p2p.transport.Transport').setLevel(logging.INFO)

    gethdb = None
    if args.gethdb is not None:
        gethdb = GethDatabase(args.gethdb)

        last_block = gethdb.last_block_hash
        last_block_num = gethdb.block_num_for_hash(last_block)

        context = f'header_hash={humanize_hash(last_block)} block_number={last_block_num}'
        logger.info(f'found geth chain tip: {context}')

    loop = asyncio.get_event_loop()
    loop.run_until_complete(connect_all(args.enodes, gethdb))
    loop.close()



if __name__ == "__main__":
    main()
