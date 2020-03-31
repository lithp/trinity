#!/usr/bin/env python

import delegator
import re
import sys


"""
I think I took the wrong route here.
1. Load all the lines into memory
2. Parse the lines into records. Drop unrecognized lines.
3. Write your functions as a bunch of reductions over the input
4. Suck it up and write SQL by hand.
   itertoolz gives you groupby!
"""


def run_p(pipeline):
    result = []
    for line in pipeline.out.split('\n'):
        if line == '':
            continue
        result.append(line.strip())
    return result


def run(cmd):
    result = []
    for line in delegator.chain(cmd).out.split('\n'):
        if line == '':
            continue
        result.append(line.strip())
    return result


def connected_peers():
    return run("grep 'peer_count' monitor_output.12 | grep ' connected' | egrep -o '<Node\(0x[0-9a-f]{6}@[0-9.]*\)>'")

def peers_which_have_sent_peers_message():
    return run("egrep 'NewBlock|NewBlockHashes' monitor_output.13 | egrep -o '<Node\(0x[0-9a-f]{6}@[0-9.]*\)>' | sort -u")

def peers_which_have_sent_ETH_block():
    return run("egrep -e NewBlock -e NewBlockHashes monitor_output.13 | egrep NewBlock | egrep 'blocknum=95[0-9]+' | egrep -o '<Node\(0x[0-9a-f]{6}@[0-9.]*\)>' | sort -u")

def peers_which_have_sent_ETC_block():
    return run("egrep -e NewBlock -e NewBlockHashes monitor_output.13 | egrep NewBlock | egrep 'blocknum=92[0-9]+' | egrep -o '<Node\(0x[0-9a-f]{6}@[0-9.]*\)>' | sort -u")

# "Transactions" also has a Node field

# There is no overlap...
eth_peers = set(peers_which_have_sent_ETH_block())
etc_peers = set(peers_which_have_sent_ETC_block())
assert len(eth_peers.intersection(etc_peers)) == 0

def emit_just_eth_lines():
    nodere = '<Node\(0x[0-9a-f]{6}@[0-9.]*\)>'
    for line in open('monitor_output.13'):
        line = line.strip()
        node = re.search(nodere, line)
        if node is None:
            continue
        node = node[0]
        if node in eth_peers:
            print(line)


def single_message_txns():
    return run("cat just_eth_peers | awk 'f{print;f=0} /Transactions: 1/{f=1}'")

def popular_single_msg_txns():
    return run_p(
        delegator.run("cat just_eth_peers")
                 .pipe("awk 'f{print;f=0} /Transactions: 1/{f=1}'")
                 .pipe("agrind '* | logfmt | count by hash, len'")
    )

def peers_which_disconnected():
    return run_p(
        delegator.run("egrep 'disconnected reason=' monitor_output.12")
                 .pipe("egrep -o '<Node\(0x[0-9a-f]{6}@[0-9.]*\)>'")
    )
disconnect_peers = set(peers_which_disconnected())

def version_string_lines():
    return run("egrep 'established connection' monitor_output.12")

def disconnect_version_strings():
    result = []
    for line in version_string_lines():
        nodere = '<Node\(0x[0-9a-f]{6}@[0-9.]*\)>'
        node = re.search(nodere, line)
        if node[0] in disconnect_peers:
            result.append(line)
    return result

def inverse_version_strings():
    result = []
    for line in version_string_lines():
        nodere = '<Node\(0x[0-9a-f]{6}@[0-9.]*\)>'
        node = re.search(nodere, line)
        if node[0] not in disconnect_peers:
            result.append(line)
    return result

node_versions = dict()
for line in version_string_lines():
    nodere = '<Node\(0x[0-9a-f]{6}@[0-9.]*\)>'
    node = re.search(nodere, line)[0]
    version = line.split(' ')[7]
    node_versions[node] = version

def nodes_which_sent_block_messages():
    return run("egrep -e NewBlock -e NewBlockHashes monitor_output.12 | egrep -o '<Node\(0x[0-9a-f]{6}@[0-9.]*\)>' | sort -u")
block_message_nodes = set(nodes_which_sent_block_messages())

#for line in nodes_which_sent_block_messages():
#    print(node_versions[line])

# now:

print('all the nodes which did not send a block message')
for line in version_string_lines():
    nodere = '<Node\(0x[0-9a-f]{6}@[0-9.]*\)>'
    node = re.search(nodere, line)[0]
    if node not in block_message_nodes:
        print(line)


# emit_just_eth_lines()

#for line in emit_just_eth_lines():
#    print(line)
