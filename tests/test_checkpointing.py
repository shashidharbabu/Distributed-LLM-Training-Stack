import os

from llmtrain.checkpointing import CheckpointManager, load_latest_checkpoint


def test_checkpoint_manifest(tmp_path):
    manager = CheckpointManager(tmp_path)
    state = {"model": {"a": 1}, "optimizer": {}, "scheduler": {}}
    manager.save(1, state, loss=0.5, is_main=True)
    manager.save(2, state, loss=0.4, is_main=True)

    latest = manager.latest_checkpoint()
    assert latest is not None
    assert latest.step == 2
    assert os.path.exists(latest.path)

    ckpt = load_latest_checkpoint(manager)
    assert ckpt is not None
    info, loaded_state = ckpt
    assert info.step == 2
    assert loaded_state["model"]["a"] == 1

    results = manager.validate_all()
    assert all(ok for _, ok, _ in results)
