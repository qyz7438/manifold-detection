from __future__ import annotations

from scripts.build_dense_local_residual_manifest import partition_candidate_ids


def test_residual_partition_is_deterministic_disjoint_and_complete():
    image_ids = list(range(1, 255))
    image_classes = {image_id: {1, 100 + image_id % 4} for image_id in image_ids}

    first = partition_candidate_ids(image_ids, image_classes=image_classes)
    second = partition_candidate_ids(list(reversed(image_ids)), image_classes=image_classes)

    assert first == second
    assert [len(first[name]) for name in ("fit", "tune", "reserve")] == [96, 32, 126]
    sets = [set(first[name]) for name in ("fit", "tune", "reserve")]
    assert not sets[0] & sets[1] and not sets[0] & sets[2] and not sets[1] & sets[2]
    assert set().union(*sets) == set(image_ids)


def test_residual_partition_balances_rare_class_and_density_tags():
    image_ids = list(range(1, 255))
    image_classes = {image_id: {1, 100 + image_id % 4} for image_id in image_ids}
    for image_id in range(1, 25):
        image_classes[image_id].add(2)

    split = partition_candidate_ids(image_ids, image_classes=image_classes)

    assert sum(2 in image_classes[image_id] for image_id in split["fit"]) >= 6
    assert sum(2 in image_classes[image_id] for image_id in split["tune"]) >= 3
    for name in ("fit", "tune"):
        density = {
            next(label for label in image_classes[image_id] if label >= 100)
            for image_id in split[name]
        }
        assert density == {100, 101, 102, 103}
