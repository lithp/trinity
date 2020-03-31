#!/usr/bin/env python

import argparse
from datetime import datetime, timedelta
import operator
import re
from typing import NamedTuple

from toolz import itertoolz, dicttoolz


class NewBlockHashes(NamedTuple):
    time: str
    remote: str
    block_hash: str
    block_num: int


class NewBlock(NamedTuple):
    time: str
    remote: str
    block_hash: str
    block_num: str
    size: int
    td: int
    transactions: int
    uncles: int


class PeerJoined(NamedTuple):
    time: str
    remote: str
    peer_count: int


class PeerLeft(NamedTuple):
    time: str
    remote: str
    peer_count: int


def parse_time(line):
    time_segment = ' '.join(line.split(' ')[:2]).replace(',', '.')
    try:
        return datetime.fromisoformat(time_segment)
    except ValueError:
        print(f'error during parse: {line}')
        raise


def parse_line(line):
    if 'NewBlockHashes' in line or 'NewBlock' in line:
        time = parse_time(line)
        if 'NewBlockHashes' in line:
            return NewBlockHashes(
                time=time,
                remote=re.search('remote=([^ ]*)', line).groups()[0],
                block_hash=re.search('hash=([0-9a-f]+)', line).groups()[0],
                block_num=int(re.search('blocknum=([0-9]+)', line).groups()[0]),
            )
        if 'NewBlock' in line:
            return NewBlock(
                time=time,
                remote=re.search('remote=([^ ]*)', line).groups()[0],
                block_hash=re.search('hash=([0-9a-f]+)', line).groups()[0],
                block_num=int(re.search('blocknum=([0-9]+)', line).groups()[0]),
                size=int(re.search('size=([0-9]+)', line).groups()[0]),
                td=int(re.search('td=([0-9]+)', line).groups()[0]),
                transactions=int(re.search('transactions=([0-9]+)', line).groups()[0]),
                uncles=int(re.search('uncles=([0-9]+)', line).groups()[0]),
            )
    if ' connected' in line and 'peer_count' in line:
        return PeerJoined(
            time=parse_time(line),
            remote=re.search('remote=([^ ]*)', line).groups()[0],
            peer_count=int(re.search('peer_count=([0-9]+)', line).groups()[0]),
        )
    if ' disconnected' in line and 'peer_count' in line:
        return PeerLeft(
            time=parse_time(line),
            remote=re.search('remote=([^ ]*)', line).groups()[0],
            peer_count=int(re.search('peer_count=([0-9]+)', line).groups()[0]),
        )
    return None


def lines(filename):
    with open(filename) as f:
        for line in f:
            yield line.strip()


def parsed(lines):
    for line in lines:
        record = parse_line(line)
        if record is not None:
            yield record


NODERE = '<Node\(0x[0-9a-f]{6}@[0-9.]*\)>'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('logfile')
    args = parser.parse_args()

    print(f'inspecting log: {args.logfile}')
    messages = list(parsed(lines(args.logfile)))

    by_msg_type = itertoolz.groupby(key=type, seq=messages)
    joined_messages = by_msg_type[PeerJoined]
    joined_peers = set(msg.remote for msg in joined_messages)

    block_anns = by_msg_type[NewBlock] + by_msg_type[NewBlockHashes]
    announced_peers = set(msg.remote for msg in block_anns)

#    print(f"{len(announced_peers)} peers announced blocks")
#    return

    joined_by_peer = itertoolz.groupby(
        key=operator.attrgetter('remote'), seq=joined_messages
    )
    peer_to_first_join = dicttoolz.valmap(
        lambda msgs: itertoolz.first(sorted(msgs, key=operator.attrgetter('time'))),
        joined_by_peer,
    )

    announce_by_peer = itertoolz.groupby(
        key=operator.attrgetter('remote'), seq=block_anns
    )
    peer_to_first_announce = dicttoolz.valmap(
        lambda msgs: itertoolz.first(sorted(msgs, key=operator.attrgetter('time'))),
        announce_by_peer,
    )

    for remote, first_announce in peer_to_first_announce.items():
        first_join = peer_to_first_join[remote]
        if first_announce.time - first_join.time > timedelta(minutes=2):
            print(f'{first_join} {first_announce}')
    return

    # for each announced peer:
      # find when we first connected to it
      # find when the first block ann arrived
      # accept the ones where there's more than a minute's difference

    print('joined={} announced={} diff={}'.format(
        len(joined_peers),
        len(announced_peers),
        len(joined_peers - announced_peers),
    ))

    print('first few:')
    for peer in itertoolz.take(10, joined_peers - announced_peers):
        print(f'- {peer}')

#    count_messages = dicttoolz.valmap(
#        itertoolz.first, by_msg_type
        # itertoolz.count, by_msg_type
#    )

#    print(
#        f"msg_types: %", count_messages
#    )

    # to count: itertoolz.count

if __name__ == '__main__':
    main()
