"""
Temporal state transition and bin accumulation logic for simulation days.

Manages the evolution of bin fill levels and determines optimal collection
schedules based on predicted overflow. Provides utilities for updating bin
states after collection and identifying 'must-go' candidates for upcoming
planning cycles.
"""

import numpy as np

from logic.src.constants.routing import MAX_CAPACITY_PERCENT


def should_bin_be_collected(current_fill_level, accumulation_rate):
    """
    Check if a bin's fill level will exceed MAX_CAPACITY_PERCENT% by the next day.

    Args:
        current_fill_level (float): Current percentage full.
        accumulation_rate (float): Daily fill rate.

    Returns:
        bool: True if overflow is predicted, None otherwise.
    """
    if current_fill_level + accumulation_rate >= MAX_CAPACITY_PERCENT:
        return True


def add_bins_to_collect(
    binsids,
    next_collection_day,
    must_go_bins,
    current_fill_levels,
    accumulation_rates,
    current_collection_day,
):
    """
    Predictively identify bins that will overflow before the next planned
    collection day and add them to the must-collect list.

    Args:
        binsids (List[int]): All relevant bin IDs.
        next_collection_day (int): Target future day.
        must_go_bins (List[int]): Current set of mandatory collections.
        current_fill_levels (Dict/np.ndarray): State.
        accumulation_rates (Dict/np.ndarray): Growth rate.
        current_collection_day (int): Today.

    Returns:
        List[int]: Updated must-collect list.
    """
    for i in binsids:
        if i in must_go_bins:
            continue
        else:
            for j in range(current_collection_day + 1, next_collection_day):
                if current_fill_levels[i] + j * accumulation_rates[i] >= MAX_CAPACITY_PERCENT:
                    must_go_bins.append(i)
                    break
    return must_go_bins


def update_fill_levels_after_first_collection(binsids, must_go_bins, current_fill_levels):
    """
    Simulate bin emptying after collection by resetting fill levels to zero.

    Args:
        binsids (List[int]): All bins.
        must_go_bins (List[int]): Bins that were collected.
        current_fill_levels (np.ndarray): State to update.

    Returns:
        np.ndarray: Updated fill levels.
    """
    current_fill_levels = current_fill_levels.copy()
    for i in binsids:
        if i in must_go_bins:
            current_fill_levels[i] = 0
    return current_fill_levels


def initialize_lists_of_bins(binsids):
    """
    Initialize a zeroed list for tracking next collection days.

    Args:
        binsids (List[int]): Bins to track.

    Returns:
        List[int]: Zero-initialized list.
    """
    next_collection_days = []
    for _i in range(0, len(binsids)):
        next_collection_days.append(0)
    return next_collection_days


def calculate_next_collection_days(must_go_bins, current_fill_levels, accumulation_rates, binsids):
    """
    Calculate how many days remain until each must-go bin overflows again.

    Args:
        must_go_bins (List[int]): Bins to analyze.
        current_fill_levels (np.ndarray): Current state.
        accumulation_rates (np.ndarray): Growth rates.
        binsids (List[int]): Total bins.

    Returns:
        List[int]: Days until next overflow per bin.
    """
    next_collection_days = initialize_lists_of_bins(binsids)
    temporary_fill_levels = current_fill_levels.copy()
    for i in must_go_bins:
        current_day = 0
        while temporary_fill_levels[i] < MAX_CAPACITY_PERCENT:
            temporary_fill_levels[i] = temporary_fill_levels[i] + accumulation_rates[i]
            current_day = current_day + 1
        next_collection_days[i] = current_day  # assuming collection happens at the beginning of the day
    return next_collection_days


def get_next_collection_day(must_go_bins, current_fill_levels, accumulation_rates, binsids):
    """
    Determine the minimum number of days until the first bin in the set overflows.

    Args:
        must_go_bins (List[int]): Target set.
        current_fill_levels (np.ndarray): State.
        accumulation_rates (np.ndarray): Daily rates.
        binsids (List[int]): Bin IDs.

    Returns:
        int: Minimum days to next overflow.
    """
    next_collection_days = calculate_next_collection_days(
        must_go_bins, current_fill_levels, accumulation_rates, binsids
    )
    next_collection_days_array = np.array(next_collection_days)
    next_collection_day = np.min(next_collection_days_array[np.nonzero(next_collection_days_array)])
    return next_collection_day
