import plyvel
import struct
import argparse

import rlp

from eth.rlp.accounts import Account
from eth.rlp.headers import BlockHeader
from trie.constants import (
    NODE_TYPE_BLANK,
    NODE_TYPE_BRANCH,
    NODE_TYPE_EXTENSION,
    NODE_TYPE_LEAF,
    BLANK_NODE_HASH,
)
from trie.utils.nodes import (
    decode_node,
    get_node_type
)


HEADER_PREFIX = b'h'
NUM_SUFFIX = b'n'


def int_to_prefix(num: int):
    # big-endian 8-byte unsigned int
    return struct.pack('>Q', num)


def get_children(nodehash, node):
    node_type = get_node_type(node)
    if node_type == NODE_TYPE_BRANCH:
        if node[16] != b'':
            raise Exception(f'[{nodehash}] unexpected node! {node}')
        return node[:16]
    elif node_type == NODE_TYPE_LEAF:
        # print(f'[{nodehash}] found leaf: {node}')
        return []
    elif node_type == NODE_TYPE_EXTENSION:
        return [node[1]]
    else:
        raise Exception(f'[{node}] unknown node type {node_type}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-db', type=str, required=True)
    parser.add_argument('-block', type=int, required=True)
    args = parser.parse_args()

    db = plyvel.DB(
        args.db,
        create_if_missing=False,
        error_if_exists=False,
        max_open_files=16
    )

    blockPrefix = HEADER_PREFIX + int_to_prefix(args.block)

    blockHashKey = blockPrefix + NUM_SUFFIX
    blockHash = db.get(blockHashKey)

    if not blockHash:
        raise Exception(f'could not find hash for block {args.block}')

    headerRlp = db.get(blockPrefix + blockHash)

    header = rlp.decode(headerRlp, sedes=BlockHeader)
    root = header.state_root
    print(f"retrieved header: {header} with stateroot: {root.hex()}")

    # Now that we have the state root, start traversing it?

    level = 0
    new_nodes = [root]
    overhead_len = 0
    account_len = 0

    while True:
        if not len(new_nodes):
            break
        cur_nodes, new_nodes = new_nodes, []

        for nodehash in cur_nodes:
            node_rlp = db.get(nodehash)
            if not node_rlp:
                raise Exception(f'was unable to fetch node {nodehash.hex()}')
            node = decode_node(node_rlp)

            node_type = get_node_type(node)
            if node_type == NODE_TYPE_LEAF:
                account_len += len(node_rlp)
                account = rlp.decode(node[1], sedes=Account)
                if account.storage_root != BLANK_NODE_HASH:
                    print(f'found storage_root: {account.storage_root.hex()}')
                    new_nodes.append(account.storage_root)
            else:
                overhead_len += len(node_rlp)

            children = get_children(nodehash, node)
            non_empty_children = [child for child in children if child != b'']
            new_nodes.extend(non_empty_children)

        print(f'[{level}] overhead: {overhead_len} acct: {account_len} queued: {len(new_nodes)}')
        level += 1

