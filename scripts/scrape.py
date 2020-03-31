#!/usr/bin/env python

from lxml import html
import requests
import re
import sys
import time

example_hash = 'a2880a12772d47f6c8192ba59c13894957fb9b4a6623e062e1af193b571c12f4'

def url_for_hash(block_hash):
    return f'https://blockscout.com/eth/mainnet/blocks/0x{block_hash}/transactions'

def get_size(block_hash):
    url = url_for_hash(block_hash)
    page = requests.get(url)

    match = re.search(b'([0-9,]+) bytes', page.content)
    if match is None:
        if b'0 Transactions' in page.content:
            return None
        print(f'invalid response! hash={block_hash}')
        import pdb; pdb.set_trace()
        print(page.content)
        sys.exit(1)

    try:
        byte_count = int(match.groups()[0].replace(b',', b''))
    except:
        print(f'invalid response! hash={block_hash}')
        print(page.content)
        raise

    return byte_count

def main(logfile_name):
    with open(logfile_name) as f:
        records = [line.strip() for line in f][1:]

    for record in records:
        blocknum = int(re.search('blocknum=([0-9]+)', record).groups()[0])
        if blocknum < 9265746:
            continue
        if blocknum > 9281328:
            break
        unclehash = re.search('unclehash=([0-9a-f]+)', record).groups()[0]
        uncompressed = get_size(unclehash)
        print(f'unclehash={unclehash} uncompressed={uncompressed}')
        time.sleep(1)

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print(f'need a single argument, the file to read')
        sys.exit(1)
    main(sys.argv[1])
