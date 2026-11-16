import numpy as np
import pytest

from scsim.inventory import Inventory, Pipeline, Shipment


def test_fefo_takes_oldest_first():
    inv = Inventory(1, 1, 30)
    inv.add(0, 0, 10, expiry=12)
    inv.add(0, 0, 5, expiry=6)
    inv.add(0, 0, 7, expiry=20)
    taken = inv.remove(0, 0, 12, today=3)
    assert taken == [(6, 5), (12, 7)]
    assert inv.total[0, 0] == 10
    assert inv.qty[0, 0, 12] == 3 and inv.qty[0, 0, 20] == 7


def test_remove_too_much_raises():
    inv = Inventory(1, 1, 10)
    inv.add(0, 0, 3, expiry=5)
    with pytest.raises(ValueError):
        inv.remove(0, 0, 4, today=0)


def test_expire_only_todays_bucket():
    inv = Inventory(2, 2, 15)
    inv.add(0, 0, 4, expiry=5)
    inv.add(0, 0, 6, expiry=6)
    inv.add(1, 1, 2, expiry=5)
    inv.add(1, 0, 9, expiry=inv.never)
    wasted = inv.expire(5)
    assert wasted.tolist() == [[4, 0], [0, 2]]
    assert inv.total.tolist() == [[6, 0], [9, 0]]
    assert inv.expiring_within(6, 3).tolist() == [[6, 0], [0, 0]]  # the never-expiring bucket is far away


def test_total_matches_buckets_after_random_ops():
    rng = np.random.default_rng(0)
    inv = Inventory(3, 2, 40)
    for day in range(25):
        for _ in range(5):
            w, p = rng.integers(3), rng.integers(2)
            inv.add(w, p, int(rng.integers(1, 20)), expiry=int(day + rng.integers(0, 10)))
        for w in range(3):
            for p in range(2):
                k = int(rng.integers(0, inv.total[w, p] + 1))
                inv.remove(w, p, k, today=day)
        inv.expire(day)
        np.testing.assert_array_equal(inv.total, inv.qty.sum(axis=2))
        assert inv.qty[:, :, :day + 1].sum() == 0  # nothing expired is left on the shelf
        assert (inv.qty >= 0).all()


def test_pipeline_waits_for_closed_warehouse():
    pipe = Pipeline(2, 1)
    pipe.add(Shipment(arrival=3, warehouse=0, product=0, qty=10, expiry=20, kind="supplier"))
    pipe.add(Shipment(arrival=3, warehouse=1, product=0, qty=4, expiry=20, kind="supplier"))
    assert pipe.pop_arrivals(2, np.array([True, True])) == []
    got = pipe.pop_arrivals(3, np.array([True, False]))  # warehouse 1 closed
    assert [s.warehouse for s in got] == [0]
    assert pipe.qty.tolist() == [[0], [4]]
    got = pipe.pop_arrivals(5, np.array([True, True]))   # reopened two days later
    assert [s.warehouse for s in got] == [1]
    assert pipe.qty.sum() == 0
