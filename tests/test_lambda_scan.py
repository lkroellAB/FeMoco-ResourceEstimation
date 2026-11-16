"""
lambda_scan.py's helpers for reading a scan: the space-time volume, the best
runtime within a qubit budget, the fewest qubits at least as fast as a
runtime and the footprint crossover between two scans.
"""
import math

from resource_count import lambda_scan as ls


def rows(*points):
    '''scan rows of (physical qubits, expected days)'''
    return [dict(phys=q, expected=t) for q, t in points]


def test_space_time_volume_is_qubit_hours():
    assert ls.space_time_volume(dict(phys=10_000, expected=0.5)) == 10_000*12


def test_best_within_a_budget():
    scan = rows((100, 3.0), (200, 2.0), (300, 1.5))
    assert ls.best_within(scan, 99) == math.inf
    assert ls.best_within(scan, 100) == 3.0
    assert ls.best_within(scan, 250) == 2.0
    assert ls.best_within(scan, 1000) == 1.5


def test_fewest_as_fast():
    points = [(300, 1.0), (100, 3.0), (200, 2.0), (150, 2.5)]
    assert ls.fewest_as_fast(points, 0.5) is None
    assert ls.fewest_as_fast(points, 1.0) == (300, 1.0)        # as fast counts
    assert ls.fewest_as_fast(points, 2.4) == (200, 2.0)
    assert ls.fewest_as_fast(points, 10) == (100, 3.0)


def test_footprint_crossover():
    # b is faster up to 250 qubits; a ties it at 300 and is ahead after
    a = rows((150, 4.0), (300, 1.4), (400, 1.0))
    b = rows((100, 3.0), (200, 1.5), (250, 1.4))
    assert ls.footprint_crossover(a, b) == (250, 300)
    # b never faster: a is at least as fast from its smallest budget
    assert ls.footprint_crossover(rows((100, 1.0)), rows((200, 2.0))) == (None, 100)
    # b faster at every budget
    assert ls.footprint_crossover(rows((200, 2.0)), rows((100, 1.0))) == (200, None)
