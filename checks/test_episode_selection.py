import hashlib
import json

import pytest

from vpp2.datasets.lerobot.episode_selection import (
    load_episode_selection,
    snapshot_episode_selection,
)


@pytest.fixture
def selection(tmp_path):
    ds = tmp_path / "suite"
    (ds / "meta").mkdir(parents=True)
    (ds / "meta/episodes.jsonl").write_text("\n".join(
        json.dumps(dict(episode_index=i, length=i + 10)) for i in range(5)
    ))
    obj = dict(schema_version=1, selected_episodes=2, selected_frames=24,
               episodes_by_dataset={str(ds): [3, 1]})
    path = tmp_path / "selection.json"
    path.write_text(json.dumps(obj))
    return ds, path, obj


def test_selection_and_immutable_run_snapshot(selection, tmp_path):
    ds, path, obj = selection
    expected = hashlib.sha256(path.read_bytes()).hexdigest()
    episodes, audit, raw = load_episode_selection(path, [str(ds)], expected)
    assert episodes == {str(ds): [1, 3]}
    assert audit["episodes"] == 2 and audit["frames"] == 24
    run = tmp_path / "run"
    snapshot_episode_selection(path, [str(ds)], run, expected)
    assert (run / "episode_manifest.json").read_bytes() == raw
    snapshot_episode_selection(path, [str(ds)], run, expected)
    obj["episodes_by_dataset"][str(ds)] = [0, 4]
    path.write_text(json.dumps(obj))
    with pytest.raises(ValueError, match="SHA256"):
        load_episode_selection(path, [str(ds)], expected)
    with pytest.raises(ValueError, match="different episode manifest"):
        snapshot_episode_selection(path, [str(ds)], run)


@pytest.mark.parametrize("ids", [[1, 1], [True, 3], [-1, 3], [1, 5], [], [1.0, 3]])
def test_invalid_episode_ids(selection, ids):
    ds, path, obj = selection
    obj["episodes_by_dataset"][str(ds)] = ids
    path.write_text(json.dumps(obj))
    with pytest.raises(ValueError):
        load_episode_selection(path, [str(ds)])


def test_dataset_and_count_mismatch(selection):
    ds, path, obj = selection
    with pytest.raises(ValueError, match="match all dataset_dirs"):
        load_episode_selection(path, [str(ds), "/not/in/manifest"])
    obj["selected_frames"] += 1
    path.write_text(json.dumps(obj))
    with pytest.raises(ValueError, match="counts"):
        load_episode_selection(path, [str(ds)])
