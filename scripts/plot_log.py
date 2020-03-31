#!/usr/bin/env python

import argparse
import logging
from datetime import datetime
import re
import random
import operator
import itertools
import statistics

from typing import NamedTuple, List

from eth_utils import toolz

import matplotlib.pyplot as plt
from matplotlib import cm
import pandas as pd
import numpy as np
from scipy import stats


logger = logging.getLogger(__file__)


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


def parse_new_block_hashes(line):
    time_segment = ' '.join(line.split(' ')[:2]).replace(',', '.')
    time=datetime.fromisoformat(time_segment)
    return NewBlockHashes(
        time=time,
        remote=re.search('remote=([^ ]*)', line).groups()[0],
        block_hash=re.search('hash=([0-9a-f]+)', line).groups()[0],
        block_num=int(re.search('blocknum=([0-9]+)', line).groups()[0]),
    )


def parse_new_block(line):
    time_segment = ' '.join(line.split(' ')[:2]).replace(',', '.')
    time=datetime.fromisoformat(time_segment)
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


def block_lines(logfile):
    for line in logfile:
        line = line.strip()
        if 'NewBlockHashes' in line:
            yield parse_new_block_hashes(line)
        elif 'NewBlock' in line:
            yield parse_new_block(line)



class MergedBlock(NamedTuple):
    first: datetime
    last: datetime
    count: int
    size: int
    hash: str
    num: int


def merge_block_records(records):
    first_record = None
    last_record = None
    record_count = 0
    size = None
    for record in records:
        record_count += 1
        if first_record is None or record.time < first_record:
            first_record = record.time
        if last_record is None or record.time > last_record:
            last_record = record.time
        if size is None and hasattr(record, 'size'):
            size = record.size
    return MergedBlock(
        first=first_record,
        last=last_record,
        count=record_count,
        hash=record.block_hash,
        num=record.block_num,
        size=size
    )


def blocks(logfile_name):
    with open(logfile_name, 'r') as logfile:
        logger.info('successfully opened logfile')
        records = block_lines(logfile)

        # Here I'm just manually executing SQL queries. Better would be to have this
        # script throw everything into a sqlite database, so I can run these queries in a
        # sqlite repl.

        by_block_hash = toolz.itertoolz.groupby(
            lambda record: record.block_hash,
            records
        )

        merged_records = [
            merge_block_records(records)
            for _block_hash, records in by_block_hash.items()
        ]

        for merged in sorted(merged_records, key=lambda rec: rec.count):
            logger.info(f'{merged}')

#        for merged in sorted(merged_records, key=lambda rec: rec.size if rec.size else 0):
#            logger.info(f'{merged}')

#        for block_hash, records in by_block_hash.items():
#            logger.debug(f'{merge_block_records(records)}')

#            logger.debug(f'records for hash {block_hash}:')
#            for record in records:
#                logger.debug(f'- {record}')


class ScatterSet(NamedTuple):
    times: List[int]
    proportions: List[float]
    block_size: int


def plot(logfile_name):
    scatter_sets = []

    with open(logfile_name, 'r') as logfile:
        records = block_lines(logfile)
        by_block_hash = toolz.itertoolz.groupby(lambda record: record.block_hash, records)

        for _block_hash, record_group in by_block_hash.items():
            times = []
            proportions = []

            record_count = len(record_group)

            if record_count < 20:
                continue

            size = None
            first_time = min(rec.time for rec in record_group)
            for i, rec in enumerate(record_group):
                times.append((rec.time - first_time).microseconds)
                proportions.append(i/float(record_count))

                if size is None and hasattr(rec, 'size'):
                    size = rec.size

            scatter_sets.append(ScatterSet(
                times=times, proportions=proportions, block_size=size,
            ))

    max_time = max(
        time
        for scatter_set in scatter_sets
        for time in scatter_set.times
    )

    sizes = [sset.block_size for sset in scatter_sets]
    for size in sorted(sizes):
        logger.info(f'size: {size}')
    size_bins = np.linspace(start=min(sizes), stop=max(sizes), num=6)
    logger.info(f'bins:')
    for size in size_bins:
        logger.info(f'size: {size}')
    size_indices = np.digitize(sizes, size_bins)

    size_bin_endpoint_pairs = toolz.itertoolz.sliding_window(2, [0] + list(size_bins))
    size_bin_labels = list(map(
        lambda items: f'{int(items[0])}-{int(items[1])}',
        size_bin_endpoint_pairs
    ))
    for label in size_bin_labels:
        logger.info(f'label: {label}')

    # this is a lot of complication because in order to get the legend to work out
    # everything with the same label needs to be plotted at the same time

    # size_indicies stores which bin each element of scatter_sets is in
    ssets_by_size_bin = toolz.itertoolz.groupby(
        lambda item: item[0],
        zip(size_indices, scatter_sets),
    )

    # drop the indicies added by enumerate
    ssets_by_size_bin = toolz.dicttoolz.valmap(
        lambda items: [item[1] for item in items],
        ssets_by_size_bin,
    )

    x_ticks = np.arange(0, max_time + 100000, 100000)
    x_tick_labels = [
            f"{xtick//1000}ms" for xtick in x_ticks
    ]

    fig = plt.figure()  # an empty figure with no axes
    fig.suptitle('Block prop over time')
    plt.xlabel('ms since first NewBlock(Hashes) message')
    plt.ylabel('Proportion of announcing nodes which announced block')

    plt.xticks(x_ticks, x_tick_labels)
    plt.tick_params(axis="x", labelrotation=30)

    viridis = cm.get_cmap('viridis', len(size_bins))

    for size, ssets in ssets_by_size_bin.items():
        logger.info(f'adding set with size {size}')
        times = []
        proportions = []
        for sset in ssets:
            times.extend(sset.times)
            proportions.extend(sset.proportions)
        plt.scatter(
            times, proportions,
            color=viridis(size),
            label=size,
        )
    plt.legend(size_bin_labels, title='Block size (bytes)')

    plt.savefig("block_prop.png", dpi="figure", bbox_inches="tight")
    logger.info('figured saved to: block_prop.png')


def delay(logfile_name):
    """
    Creates a scatterplot:
    - x: the size of the block
    - y: the delay until a majority of the network knows about the block
    """
    with open(logfile_name, 'r') as logfile:
        logger.info('successfully opened logfile')
        records = list(block_lines(logfile))

    by_block_hash = toolz.itertoolz.groupby(lambda record: record.block_hash, records)

    def records_to_block_size(records):
        with_size = [
            getattr(record, 'size') for record in records
            if hasattr(record, 'size')
        ]
        if len(with_size) > 1:
            return with_size[1]
        return None

    block_hash_to_size = toolz.dicttoolz.valmap(records_to_block_size, by_block_hash)
    block_hash_to_size = toolz.dicttoolz.valfilter(
        lambda x: x is not None, 
        block_hash_to_size
    )

    blocks_with_size = list(block_hash_to_size.keys())
    by_block_hash = toolz.dicttoolz.keyfilter(
        lambda block_hash: block_hash in blocks_with_size,
        by_block_hash
    )

    def first_announcement_from_each_peer(records):
        # some nodes send us multiple announcements for the same block. No idea why.
        by_remote = toolz.itertoolz.groupby(lambda record: record.remote, records)

        time = operator.attrgetter('time')
        remote_to_first_message = toolz.dicttoolz.valmap(
            lambda records: toolz.itertoolz.first(sorted(records, key=time)),
            by_remote
        )

        return list(sorted(
            remote_to_first_message.values(),
            key=time
        ))

    block_hash_to_first_announcements = toolz.dicttoolz.valmap(
        first_announcement_from_each_peer, by_block_hash
    )

    # to prevent blocks with small sample sizes from throwing us off, we only consider
    # blocks which more than x nodes told us about

    block_hash_to_first_announcements = toolz.dicttoolz.valfilter(
        lambda records: len(records) > 10,
        block_hash_to_first_announcements
    )

    def seconds_until_majority_announced_block(records):
        majority = len(records) // 2 + 1
        majority_time = records[majority].time
        return (majority_time - records[0].time).total_seconds()

    block_hash_to_majority_delay = toolz.dicttoolz.valmap(
        seconds_until_majority_announced_block,
        block_hash_to_first_announcements
    )

    missing_keys = set(block_hash_to_majority_delay.keys()) - set(block_hash_to_size.keys())
    assert len(missing_keys) == 0

    # Now, build a bunch of (size, delay) tuples!

    sizes = []
    delays = []

    block_hash_to_size_and_delay = dict()
    for block_hash, delay in block_hash_to_majority_delay.items():
        size = block_hash_to_size[block_hash]
        block_hash_to_size_and_delay[block_hash] = (size, delay)
        sizes.append(size)
        delays.append(delay)

    fig = plt.figure()
    fig.suptitle('Block size vs majority acceptance')
    plt.xlabel('block size (bytes)')
    plt.ylabel('Seconds until most nodes announced block')

    plt.scatter(sizes, delays)
    plt.savefig("block_delay_by_size.png", dpi="figure", bbox_inches="tight")
    logger.info('figured saved to: block_delay_by_size.png')

    # For my second scatter plot, I want to filter out all the props which took more time
    # First, bin the sizes into a fwe different buckets.
    # Then, for each bucket, only plot the bottom 10% of delays within that bucket

    block_sizes = [
        size for (size, _delay) in block_hash_to_size_and_delay.values()
    ]

    size_bins = np.linspace(start=min(block_sizes), stop=max(block_sizes), num=1)
    def digitize(size):
        return np.digitize([size], size_bins)[0]

    groupby = toolz.itertoolz.groupby
    all_sizes_and_delays = list(block_hash_to_size_and_delay.values())
    size_bin_to_size_and_delays = groupby(lambda x: digitize(x[0]), all_sizes_and_delays)

    # for each size bin, I want to sort the elements by delay and take the bottom 10%
    def bottom_10_percent(size_and_delays):
        s = list(sorted(size_and_delays, key=lambda x: x[1]))
        return s[:(len(s)//2)+1]
        # return s

    size_bin_to_smallest_size_and_delays = toolz.dicttoolz.valmap(
        bottom_10_percent, size_bin_to_size_and_delays
    )

    all_sizes_and_delays = toolz.itertoolz.concat(
        size_bin_to_smallest_size_and_delays.values()
    )

    sizes, delays = [], []
    for (size, delay) in all_sizes_and_delays:
        sizes.append(size)
        delays.append(delay)

    slope, intercept, r_value, p_value, std_err = stats.linregress(sizes, delays)

    min_x, max_x = min(sizes), max(sizes)
    min_y = slope * min_x + intercept
    max_y = slope * max_x + intercept

    print(f'latency_cost(ms)={intercept*1000}, cost(ms)_per_kb={slope*1024*1000}, corr={r_value}')

    fig = plt.figure()
    fig.suptitle('Block size vs majority acceptance (bottom 50%)')
    plt.xlabel('block size (bytes)')
    plt.ylabel('Seconds until most nodes announced block')

    plt.scatter(sizes, delays)
    plt.plot([min_x, max_x], [min_y, max_y])
    plt.savefig("minimal_block_delay_by_size.png", dpi="figure", bbox_inches="tight")
    logger.info('figured saved to: minimal_block_delay_by_size.png')


def cdf(logfile_name):
    """
    x axis: time since first announcement
    y axis: percentage of nodes which have announced the block

    This is easy to graph for a single block, but the results need to be merged across all
    blocks.

    For the "average block", I want to know the chance that a randomly chosen peer has
    announced that block after x seconds. How will I get a different answer than if I ask
    for the median block?

    Average block:
    1. Build the cdf for every block
    2. For each time you want to sample (linspace)
       - Find the percentages from each block and take the average

    Median block:
    1. Build the cdf for every block
    2. For each time you want to sample (linspace)
       - Find the percentages from each block and take the median

    Also, draw some lines at 10%, 50%, and 90%, to make it easy to see where they intersect
    """
    with open(logfile_name, 'r') as logfile:
        logger.info('successfully opened logfile')
        records = list(block_lines(logfile))

    by_block_hash = toolz.itertoolz.groupby(lambda record: record.block_hash, records)

    def first_announcement_from_each_peer(records):
        # some nodes send us multiple announcements for the same block. No idea why.
        by_remote = toolz.itertoolz.groupby(lambda record: record.remote, records)

        time = operator.attrgetter('time')
        remote_to_first_message = toolz.dicttoolz.valmap(
            lambda records: toolz.itertoolz.first(sorted(records, key=time)),
            by_remote
        )

        return list(sorted(
            remote_to_first_message.values(),
            key=time
        ))

    block_hash_to_first_announcements = toolz.dicttoolz.valmap(
        first_announcement_from_each_peer, by_block_hash
    )

    # to prevent blocks with small sample sizes from throwing us off, we only consider
    # blocks which more than x nodes told us about

    block_hash_to_first_announcements = toolz.dicttoolz.valfilter(
        lambda records: len(records) > 10,
        block_hash_to_first_announcements
    )

    def compute_propagation_cdf(records):
        """
        Accepts a set of records and returns a function(time) -> percentage
        """

        # 1. build the cdf
        # we want a list of tuples: (seconds_since_arrival, percentage of nodes)
        # there's a cute functional way to do this but w/e

        result = []
        total_records = len(records)
        first_time = records[0].time
        for i, record in enumerate(records):
            result.append(((record.time - first_time).total_seconds(), i/total_records))

        # 2. build the function
        # given a time we slide over the records looking for where we belong
        # (secs_1, percentage_1) < seconds < (secs_2, percentage_2) -> percentage_1
        # secs_0 is always 0, so we cant go under
        # secs might be greater than the largest value, in which case return 1

        final_time = result[-1][0]

        def propagation_cdf(seconds):
            if seconds == 0:
                return result[0][1]
            assert seconds > 0
            matching = list(itertools.takewhile(
                lambda tup: tup[0] < seconds,
                result,
            ))
            return matching[-1][1]
        return (propagation_cdf, final_time)

    block_hash_to_cdf = toolz.dicttoolz.valmap(
        compute_propagation_cdf,
        block_hash_to_first_announcements
    )

    cdfs_and_final_times = list(block_hash_to_cdf.values())
    final_times = [final_time for (_cdf_func, final_time) in cdfs_and_final_times]
    final_time = max(final_times)

    cdfs = [cdf for (cdf, _final_time) in cdfs_and_final_times]
    def average_cdf(time):
        return statistics.mean(
            cdf(time) for cdf in cdfs
        )

    x_values = np.linspace(start=0, stop=15, num=200)

    # There has to be a cleaner way of inverting this function
    # surely numpy/scipy has something?
    p_10, p_50, p_90 = None, None, None
    y_values = []
    for x_value in x_values:
        average_reporting_node_percentage = average_cdf(x_value)
        if p_10 is None and average_reporting_node_percentage > 0.1:
            p_10 = x_value
        if p_50 is None and average_reporting_node_percentage > 0.5:
            p_50 = x_value
        if p_90 is None and average_reporting_node_percentage > 0.9:
            p_90 = x_value
        y_values.append(average_reporting_node_percentage)

    print(f'p_10(ms)={p_10*1000} p_50(ms)={p_50*1000} p_90(ms)={p_90*1000}')

    fig = plt.figure()
    fig.suptitle('Block propagation')
    plt.xlabel('time (seconds)')
    plt.ylabel('percentage of peers which announced block')

    plt.plot(x_values, y_values)
    plt.savefig("block_cdf.png", dpi="figure", bbox_inches="tight")
    logger.info('figured saved to: block_cdf.png')


class BlockSize(NamedTuple):
    time: datetime
    day: int
    blocknum: int
    uncompressed: int
    compressed: int
    uncles: int
    gas: int


class UncleBlock(NamedTuple):
    blocknum: int
    uncompressed: int


def parse_uncle_block(line):
    size_match = re.search('uncompressed=([0-9]+)', line)
    if size_match is None:
        return None
    return UncleBlock(
        blocknum=int(re.search('blocknum=([0-9]+)', line).groups()[0]),
        uncompressed=int(size_match.groups()[0]),
    )


def parse_block_size(line):
    return BlockSize(
        time=datetime.utcfromtimestamp(int(re.search('time=([0-9]+)', line).groups()[0])),
        #time=int(re.search('time=([0-9]+)', line).groups()[0]),
        day=int(re.search('day=([0-9]+)', line).groups()[0]),
        blocknum=int(re.search('blocknum=([0-9]+)', line).groups()[0]),
        uncompressed=int(re.search('uncompressed=([0-9]+)', line).groups()[0]),
        compressed=int(re.search(' compressed=([0-9]+)', line).groups()[0]),
        uncles=int(re.search('uncles=([0-9]+)', line).groups()[0]),
        gas=int(re.search('gas_used=([0-9]+)', line).groups()[0]),
    )


def plot_gas_usage_against_block_size(records, target):
    records = [
        record for record in records
        if record.blocknum > 9265754 and record.blocknum < 9271633
    ]

    sizes = []
    gas_usages = []
    for record in records:
        sizes.append(record.compressed)
        gas_usages.append(record.gas)

    slope, intercept, r_value, p_value, std_err = stats.linregress(gas_usages, sizes)
    min_x, max_x = min(gas_usages), max(gas_usages)
    min_y = slope * min_x + intercept
    max_y = slope * max_x + intercept
    print(f'intercept={intercept}, slope={slope}, corr={r_value}')

    fig = plt.figure()
    plt.scatter(gas_usages, sizes)
    plt.plot([min_x, max_x], [min_y, max_y])
    plt.savefig(target, dpi="figure", bbox_inches="tight")
    logger.info(f'figured saved to: {target}')


def plot_uncompressed_against_compressed_size(records, target):
    records = [
        record for record in records
        if record.blocknum > 9265754 and record.blocknum < 9271633
    ]

    uncompressed = list(map(operator.attrgetter('uncompressed'), records))
    compressed = list(map(operator.attrgetter('compressed'), records))

    slope, intercept, r_value, p_value, std_err = stats.linregress(
        uncompressed, compressed
    )
    min_x, max_x = min(uncompressed), max(uncompressed)
    min_y = slope * min_x + intercept
    max_y = slope * max_x + intercept
    print(f'intercept={intercept}, slope={slope}, corr={r_value}')

    fig = plt.figure()
    fig.suptitle('Compressed vs uncompressed size')
    plt.xlabel('Uncompressed block size (bytes)')
    plt.ylabel('Compressed block size (bytes)')
    plt.scatter(uncompressed, compressed)
    plt.plot([min_x, max_x], [min_y, max_y], label="3950+0.44x")
    plt.legend()
    plt.savefig(target, dpi="figure", bbox_inches="tight")
    logger.info(f'figured saved to: {target}')


def plot_uncle_against_sibling_sizes(canon_records, uncle_records, target):
    block_num_to_uncles = toolz.itertoolz.groupby(
        operator.attrgetter('blocknum'), uncle_records
    )
    block_num_to_canons = toolz.itertoolz.groupby(
        operator.attrgetter('blocknum'), canon_records
    )
    multiple_canons = toolz.dicttoolz.valfilter(
        lambda x: len(x) > 1, block_num_to_canons
    )
    assert len(multiple_canons) == 0
    block_num_to_canons = toolz.dicttoolz.valmap(
        lambda x: x[0], block_num_to_canons
    )

    canon_sizes = []
    uncle_sizes = []
    for block_num in block_num_to_uncles.keys():
        canonical_block = block_num_to_canons[block_num]
        uncles = block_num_to_uncles[block_num]

        for uncle in uncles:
            canon_sizes.append(canonical_block.uncompressed / 1024)
            uncle_sizes.append(uncle.uncompressed / 1024)


    slope, intercept, r_value, p_value, std_err = stats.linregress(canon_sizes, uncle_sizes)
    min_x, max_x = min(canon_sizes), max(canon_sizes)
    min_y = slope * min_x + intercept
    max_y = slope * max_x + intercept
    print(f'intercept={intercept}, slope={slope}, corr={r_value}')

    fig = plt.figure()
    plt.plot([min_x, max_x], [min_y, max_y])
    plt.scatter(canon_sizes, uncle_sizes)
    # plt.axis(xmin=0, xmax=100, ymin=0, ymax=100)
    plt.savefig(target, dpi="figure", bbox_inches="tight")
    logger.info(f'figured saved to: {target}')

    return


    # 1. For each set of siblings, how much bigger are the uncles?
    ratios = []
    for block_num in sorted(block_num_to_uncles.keys()):
        canonical_block = block_num_to_canons[block_num]
        uncles = block_num_to_uncles[block_num]
        uncle_size = statistics.mean(uncle.uncompressed for uncle in uncles)
        ratio = uncle_size - canonical_block.uncompressed
        print(
            f'num={block_num} '
            f'canon={canonical_block.uncompressed} '
            f'uncles={len(uncles)} '
            f'uncle={uncle_size} '
            f'ratio={ratio}'
        )
        ratios.append(ratio)
    print(f'average ratio {statistics.mean(ratios)}')

    fig = plt.figure()
    plt.boxplot(ratios, vert=False)
    plt.savefig(target, dpi="figure", bbox_inches="tight")
    logger.info(f'figured saved to: {target}')
    pass


def plot_block_size_distributions(canon_records, uncle_records, target):
    canon_sizes = [canon.uncompressed for canon in canon_records]
    uncle_sizes = [uncle.uncompressed for uncle in uncle_records]
    print(f'canon={statistics.mean(canon_sizes)} uncle={statistics.mean(uncle_sizes)}')

    min_size = min(min(canon_sizes), min(uncle_sizes))
    max_size = max(max(canon_sizes), max(canon_sizes))
    bins = np.linspace(min_size, max_size, 7)

    canon_hist, canon_bins = np.histogram(canon_sizes, bins=bins)
    uncle_hist, uncle_bins = np.histogram(uncle_sizes, bins=bins)

    # normalize the buckets
    canon_hist = canon_hist / sum(canon_hist)
    uncle_hist = uncle_hist / sum(uncle_hist)

    size_bin_endpoint_pairs = toolz.itertoolz.sliding_window(2, bins)
    size_bin_labels = list(map(
        lambda items: f'{int(items[0])}-{int(items[1])}',
        size_bin_endpoint_pairs
    ))

    x = np.arange(len(size_bin_labels))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, canon_hist, width, label='Canonical Blocks')
    rects2 = ax.bar(x + width/2, uncle_hist, width, label='Uncled Blocks')

    ax.set_xlabel('Block size (bytes)')
    ax.set_title('Size distribution of each block type')
    ax.set_xticks(x)
    ax.set_xticklabels(size_bin_labels)
    ax.tick_params(axis="x", labelrotation=30)
    ax.legend()

    def autolabel(rects, xoffset):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{:.2%}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(xoffset, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(rects1, xoffset=-5)
    autolabel(rects2, xoffset=5)

    fig.tight_layout()

    plt.savefig(target, dpi="figure", bbox_inches="tight")
    logger.info(f'figured saved to: {target}')


def plot_uncle_probability_per_block_size(canon_records, uncle_records, target):
    """
    Give me a line graph. For each block size, how likely are blocks to be uncled?
    """
    un = operator.attrgetter('uncompressed')

    canon_records = [
        record for record in canon_records
        if record.blocknum >= 9265746 and record.blocknum <= 9273973
    ]
    uncle_records = [
        record for record in uncle_records
        if record.blocknum >= 9265746 and record.blocknum <= 9273973
    ]

#    canon_records = random.sample(canon_records, len(uncle_records))

#    canon_records = [r for r in canon_records if r.uncompressed > 1024*512]
#    uncle_records = [r for r in uncle_records if r.uncompressed > 1024*512]

    all_sizes = list(map(un, canon_records)) + list(map(un, uncle_records))
    min_size, max_size = min(all_sizes), max(all_sizes)+1
    bins = np.linspace(min_size, max_size, 5)
    bins = [0, 1024*128, 1024*256, 1024*512, 1024*1024, 1024*1024*2]

    size_bin_endpoint_pairs = toolz.itertoolz.sliding_window(2, bins)
    size_bin_labels = list(map(
        lambda items: f'{int(items[0])}-{int(items[1])}',
        size_bin_endpoint_pairs
    ))
    for label in size_bin_labels:
        logger.info(f'label: {label}')

    def digitize(size):
        return np.digitize([size], bins)[0]

    d = toolz.functoolz.compose(digitize, un)
    size_to_canons = toolz.itertoolz.groupby(d, canon_records)
    size_to_uncles = toolz.itertoolz.groupby(d, uncle_records)

    for i in range(1, len(bins)+1):
        canon_count = len(size_to_canons.get(i, []))
        uncle_count = len(size_to_uncles.get(i, []))
        if uncle_count + canon_count == 0:
            print(f'bin={i} canon={canon_count} uncle={uncle_count}')
            continue
        ratio = uncle_count / (uncle_count + canon_count)
        print(f'bin={size_bin_labels[i-1]} canon={canon_count} uncle={uncle_count} r={ratio:.2%}')

    canon_sizes = [canon.uncompressed for canon in canon_records]
    uncle_sizes = [uncle.uncompressed for uncle in uncle_records]

    fig = plt.figure()
    plt.boxplot([canon_sizes, uncle_sizes], vert=False, labels=['canon', 'uncle'])
    plt.savefig(target, dpi="figure", bbox_inches="tight")


def plot_size_distribution(canon_records, target):
    per_day = toolz.itertoolz.groupby(operator.attrgetter('day'), canon_records)
    day_to_block_count = toolz.dicttoolz.valmap(len, per_day)

    # So, each day 90% of blocks were under 70k in compressed size
    print(f'message per day: ', day_to_block_count)

    def messages_under_70k(records):
        under_70k = [record for record in records if record.uncompressed < 70000]
        return len(under_70k) / len(records)
    day_to_70k = toolz.dicttoolz.valmap(messages_under_70k, per_day)
    print(f'messages under 70k per day:', day_to_70k)

    median_each_day = toolz.dicttoolz.valmap(
        lambda x: statistics.median(r.uncompressed for r in x),
        per_day
    )
    print(f'median per day: ', median_each_day)

    fig = plt.figure()

    # Logspace
    plt.subplot(212)
    bins = np.logspace(0, np.log10(1500), 30)
    for day in [11, 12]:
        day_records = per_day[day]
        sizes = list(map(operator.attrgetter('uncompressed'), day_records))
        sizes = [size / 1024 for size in sizes]
        #sizes = list(filter(lambda size: size < 60000, sizes))
        plt.hist(sizes, bins, alpha=0.5, label=f'Jan {day}')
    plt.xscale('log')
    plt.legend(title='Day')
    plt.xlabel('Block size (KB)')

    # Linspace
    plt.subplot(211)
    bins = np.linspace(0, 100, 30)
    for day in [11, 12]:
        day_records = per_day[day]
        sizes = list(map(operator.attrgetter('uncompressed'), day_records))
        sizes = [size / 1024 for size in sizes]
        #sizes = list(filter(lambda size: size < 60000, sizes))
        plt.hist(sizes, bins, alpha=0.5, label=f'Jan {day}')
    plt.legend(title='Day')
    
    plt.title('Block size distribution before and during experiment')
    plt.savefig("starkware_size_dist.png", dpi="figure", bbox_inches="tight")
    logger.info('figured saved to: starkware_size_dist.png')


def plot_size_to_uncle_size(canon_records, target):
    block_window = toolz.itertoolz.sliding_window(1, records)
    def average_sizes(window):
        middle = window[len(window)//2]
        return middle._replace(
            compressed=statistics.mean(item.compressed for item in window),
        )
    averaged_blocks = [average_sizes(window) for window in block_window]

    sizes = []
    uncle_counts = []
    for block in averaged_blocks:
        uncle_counts.append(block.uncles)
        sizes.append(block.compressed)

    slope, intercept, r_value, p_value, std_err = stats.linregress(sizes, uncle_counts)
    min_x, max_x = min(sizes), max(sizes)
    min_y = slope * min_x + intercept
    max_y = slope * max_x + intercept
    print(f'intercept={intercept}, slope={slope*1024*1024}, corr={r_value}')

    fig = plt.figure()
    plt.scatter(sizes, uncle_counts)
    plt.plot([min_x, max_x], [min_y, max_y])
    plt.savefig("starkware_size_v_uncle_count.png", dpi="figure", bbox_inches="tight")
    logger.info('figured saved to: starkware_size_v_uncle_count.png')


def plot_size_to_brother_count(canon_records, uncle_records, target):
    un = operator.attrgetter('uncompressed')

    canon_records = [
        record for record in canon_records
        if record.blocknum >= 9265746 and record.blocknum <= 9273973
    ]
    uncle_records = [
        record for record in uncle_records
        if record.blocknum >= 9265746 and record.blocknum <= 9273973
    ]

    block_num_to_uncles = toolz.itertoolz.groupby(
        operator.attrgetter('blocknum'), uncle_records
    )

    block_sizes = []
    brother_counts = []
    for canon_record in canon_records:
        brother_count = len(block_num_to_uncles.get(canon_record.blocknum, []))
        block_sizes.append(un(canon_record))
        brother_counts.append(brother_count)

    fig = plt.figure()
    plt.scatter(block_sizes, brother_counts)
    plt.savefig(target, dpi="figure", bbox_inches="tight")
    logger.info(f'figured saved to: {target}')


def plot_size_to_brother_count_binned(canon_records, uncle_records, target):
    un = operator.attrgetter('uncompressed')

    canon_records = [
        record for record in canon_records
        if record.blocknum >= 9265746 and record.blocknum <= 9273973
    ]
    uncle_records = [
        record for record in uncle_records
        if record.blocknum >= 9265746 and record.blocknum <= 9273973
    ]

    block_num_to_uncles = toolz.itertoolz.groupby(
        operator.attrgetter('blocknum'), uncle_records
    )

    bins = np.linspace(0,1500, 8)
    # bins = np.logspace(0, np.log10(1500), 6)
    def digitize(size):
        return np.digitize([size], bins)[0]
    size_bin_to_canon_blocks = toolz.itertoolz.groupby(
        lambda rec: digitize(un(rec) / 1024),
        canon_records
    )

    size_bin_endpoint_pairs = toolz.itertoolz.sliding_window(2, list(bins))
    size_bin_labels = list(map(
        lambda items: f'{int(items[0])}-{int(items[1])}',
        size_bin_endpoint_pairs
    ))

    def brother_count(canon_record):
        return len(block_num_to_uncles.get(canon_record.blocknum, []))

    def average_brother_count(records):
        return statistics.mean(brother_count(r) for r in records)

    for size_bin in range(0, len(bins)-1):
        recs = size_bin_to_canon_blocks.get(size_bin+1, [])
        brothers = sum(brother_count(r) for r in recs)
        avg_bro_count = average_brother_count(recs)
        print(
            f'bin={size_bin_labels[size_bin]} ',
            f'blocks={len(recs)} ',
            f'brothers={brothers} ',
            f'avg_bro_count={avg_bro_count} '
        )

    size_bin_to_avg_bro_count = toolz.dicttoolz.valmap(
        average_brother_count,
        size_bin_to_canon_blocks
    )

    for size, bro_count in sorted(size_bin_to_avg_bro_count.items()):
        print(size, bro_count)

    weights = []
    for i in range(0, len(bins)-1):
        weights.append(size_bin_to_avg_bro_count.get(i+1, 0))

    fig = plt.figure()
    plt.subplot(212)
    plt.hist(bins[:-1], bins, weights=weights)
    plt.xlabel('Block size (KB)')
    plt.ylabel('Average brothers per block')
    # plt.xscale('log')
    #plt.savefig(target, dpi="figure", bbox_inches="tight")
    #logger.info(f'figured saved to: {target}')

    block_sizes = []
    brother_counts = []
    for canon_record in canon_records:
        brother_count = len(block_num_to_uncles.get(canon_record.blocknum, []))
        block_sizes.append(un(canon_record) / 1024)
        brother_counts.append(brother_count)

    plt.subplot(211)
    plt.scatter(block_sizes, brother_counts)
    plt.axis(xmin=-100, xmax=1600, ymin=-0.25, ymax=3.25)
    plt.ylabel('Brother count')

    plt.title('Brian size for brother count')
    plt.savefig(target, dpi="figure", bbox_inches="tight")
    logger.info(f'figured saved to: {target}')



def starkware(logfile_name, uncles_logfile):
    with open(logfile_name) as f:
        lines = list(f)
        lines = lines[1:]
        canon_records = [parse_block_size(line.strip()) for line in lines]

    with open(uncles_logfile) as f:
        uncle_records = list(filter(bool, [parse_uncle_block(l.strip()) for l in f]))

    plot_size_to_brother_count_binned(
        canon_records, uncle_records, "starkware_size_to_brother_count.png"
    )

    return

    plot_size_to_brother_count(
        canon_records, uncle_records, "starkware_size_to_brother_count.png"
    )

    return

    plot_uncle_against_sibling_sizes(
        canon_records, uncle_records, "starkware_uncle_size_ratio_hist.png"
    )

    return

    plot_size_to_uncle_size(
        canon_records,
        'starkware_size_v_uncle_size.png'
    )

    return

    plot_size_distribution(
        canon_records, 'starkware_size_dist.png'
    )

    return

    plot_uncompressed_against_compressed_size(
        canon_records, 'starkware_uncompressed_to_compressed.png'
    )

    return

    plot_uncle_probability_per_block_size(
        canon_records, uncle_records, 'starkware_probs.png'
    )

    # target = "starkware_size_dist_w_uncle.png"
    # plot_block_size_distributions(canon_records, uncle_records, target)


    # 1. Plot gas usage against compressed size, do they correlate well?
    # plot_gas_usage_against_block_size(records, 'starkware_gas_to_size_scatter.png')

    # Only look at records from Starkware blocks, hopefully that increases the correlation

    return

    per_day = toolz.itertoolz.groupby(operator.attrgetter('day'), records)
    day_to_block_count = toolz.dicttoolz.valmap(len, per_day)

    # So, each day 90% of blocks were under 70k in compressed size
    print(f'message per day: ', day_to_block_count)

    def messages_under_70k(records):
        under_70k = [record for record in records if record.compressed < 70000]
        return len(under_70k) / len(records)
    day_to_70k = toolz.dicttoolz.valmap(messages_under_70k, per_day)
    print(f'messages under 70k per day:', day_to_70k)

    fig = plt.figure()
    bins = np.linspace(0, 60000, 60)
    days = []
    for day, day_records in per_day.items():
        sizes = list(map(operator.attrgetter('compressed'), day_records))
        sizes = list(filter(lambda size: size < 60000, sizes))
        plt.hist(sizes, bins, alpha=0.25, label=f'Jan {day}')

    plt.legend(title='Day')
    plt.savefig("starkware_size_dist.png", dpi="figure", bbox_inches="tight")
    logger.info('figured saved to: starkware_size_dist.png')

    # Okay, that was per day. Now let's do per-hour
    hour = lambda x: (x.year, x.month, x.day, x.hour)
    per_hour = toolz.itertoolz.groupby(lambda x: hour(x.time), records)

    sizes = []
    uncle_rates = []
    for hour, blocks in per_hour.items():
        uncles = sum(record.uncles for record in blocks)
        avg_size = statistics.mean(record.compressed for record in blocks)
        uncle_rate = uncles/len(blocks)
        print(f'hour={hour} blocks={len(blocks)} uncles={uncles} size={avg_size}')

        sizes.append(avg_size)
        uncle_rates.append(uncle_rate)

    fig = plt.figure()
    plt.scatter(sizes, uncle_rates)
    plt.savefig("starkware_size_v_uncle_rate_hourly.png", dpi="figure", bbox_inches="tight")
    logger.info('figured saved to: starkware_size_v_uncle_rate_hourly.png')

    # Okay, try again but with buckets 1000 blocks wide...

    per_bucket = toolz.itertoolz.groupby(lambda r: r.blocknum // 1000, records)

    sizes = []
    uncle_rates = []
    for bucket, blocks in per_bucket.items():
        uncles = sum(record.uncles for record in blocks)
        avg_size = statistics.mean(record.compressed for record in blocks)
        uncle_rate = uncles/len(blocks)
        print(f'bucket={bucket} blocks={len(blocks)} uncles={uncles} size={avg_size}')

        sizes.append(avg_size)
        uncle_rates.append(uncle_rate)

    slope, intercept, r_value, p_value, std_err = stats.linregress(sizes, uncle_rates)
    min_x, max_x = min(sizes), max(sizes)
    min_y = slope * min_x + intercept
    max_y = slope * max_x + intercept
    print(f'intercept={intercept}, slope={slope*1024*1024}, corr={r_value}')

    fig = plt.figure()
    plt.scatter(sizes, uncle_rates)
    plt.plot([min_x, max_x], [min_y, max_y])
    plt.savefig("starkware_size_v_uncle_rate_buckets.png", dpi="figure", bbox_inches="tight")
    logger.info('figured saved to: starkware_size_v_uncle_rate_buckets.png')

    # Okay, but try again and use a sliding window, bucket by the average block size in
    # that window.

    block_window = toolz.itertoolz.sliding_window(1, records)
    def average_sizes(window):
        middle = window[len(window)//2]
        return middle._replace(
            compressed=statistics.mean(item.compressed for item in window),
        )
    averaged_blocks = [average_sizes(window) for window in block_window]

    sizes = []
    uncle_counts = []
    for block in averaged_blocks:
        uncle_counts.append(block.uncles)
        sizes.append(block.compressed)

    slope, intercept, r_value, p_value, std_err = stats.linregress(sizes, uncle_counts)
    min_x, max_x = min(sizes), max(sizes)
    min_y = slope * min_x + intercept
    max_y = slope * max_x + intercept
    print(f'intercept={intercept}, slope={slope*1024*1024}, corr={r_value}')

    fig = plt.figure()
    plt.scatter(sizes, uncle_counts)
    plt.plot([min_x, max_x], [min_y, max_y])
    plt.savefig("starkware_size_v_uncle_count.png", dpi="figure", bbox_inches="tight")
    logger.info('figured saved to: starkware_size_v_uncle_count.png')

    return
        

    block_sizes = list(sorted(map(operator.attrgetter('compressed'), averaged_blocks)))
    bins = np.linspace(block_sizes[0], block_sizes[-1], 7)
    def digitize(size):
        return np.digitize([size], bins)[0]
    per_size = toolz.itertoolz.groupby(lambda r: digitize(r.compressed), averaged_blocks)

    sizes = []
    uncle_rates = []
    for bucket, blocks in per_size.items():
        uncles = sum(record.uncles for record in blocks)
        avg_size = statistics.mean(record.compressed for record in blocks)
        uncle_rate = uncles/len(blocks)
        print(f'bucket={bucket} blocks={len(blocks)} uncles={uncles} size={avg_size}')

        sizes.append(avg_size)
        uncle_rates.append(uncle_rate)

    slope, intercept, r_value, p_value, std_err = stats.linregress(sizes, uncle_rates)
    min_x, max_x = min(sizes), max(sizes)
    min_y = slope * min_x + intercept
    max_y = slope * max_x + intercept
    print(f'intercept={intercept}, slope={slope*1024*1024}, corr={r_value}')

    fig = plt.figure()
    plt.scatter(sizes, uncle_rates)
    plt.plot([min_x, max_x], [min_y, max_y])
    plt.savefig("starkware_size_v_uncle_rate_size_bins.png", dpi="figure", bbox_inches="tight")
    logger.info('figured saved to: starkware_size_v_uncle_rate_size_bins.png')


def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s: %(message)s'
    )

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")

    blocks_parser = subparsers.add_parser('blocks')
    blocks_parser.add_argument('logfile')

    plot_parser = subparsers.add_parser('plot')
    plot_parser.add_argument('logfile')

    delay_parser = subparsers.add_parser('delay')
    delay_parser.add_argument('logfile')

    cdf_parser = subparsers.add_parser('cdf')
    cdf_parser.add_argument('logfile')

    starkware_parser = subparsers.add_parser('starkware')
    starkware_parser.add_argument('canonical_logfile')
    starkware_parser.add_argument('uncles_logfile')

    args = parser.parse_args()

    if args.command == 'blocks':
        blocks(args.logfile)
    elif args.command == 'plot':
        plot(args.logfile)
    elif args.command == 'delay':
        delay(args.logfile)
    elif args.command == 'cdf':
        cdf(args.logfile)
    elif args.command == 'starkware':
        starkware(args.canonical_logfile, args.uncles_logfile)
    else:
        logger.error(f'unrecognized command: {args.command}')


if __name__ == '__main__':
    main()
