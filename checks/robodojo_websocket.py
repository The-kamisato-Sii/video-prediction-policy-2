"""Exercise a running full-policy server using synthetic native RoboDojo observations.

This checks the websocket adapter, not simulator execution or task success.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robodojo-root", type=Path, required=True)
    parser.add_argument("--url", default="ws://127.0.0.1:23000")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(args.robodojo_root.resolve() / "XPolicyLab"))
    from client_server.ws.model_client import WsModelClient

    obs = {
        "env_idx": 0,
        "task_instruction": "pick up the block",
        "vision": {
            camera: {"color": np.zeros((240, 320, 3), dtype=np.uint8)}
            for camera in ("cam_head", "cam_left_wrist", "cam_right_wrist")
        },
        "state": {
            f"{side}_{field}": np.array(value, dtype=np.float32)
            for side in ("left", "right")
            for field, value in (
                ("ee_pose", [0.0, 0.0, 0.3, 1.0, 0.0, 0.0, 0.0]),
                ("ee_joint_state", [0.5]),
            )
        },
    }
    with WsModelClient(
        url=args.url,
        evaluation_id="vpp2_synthetic_transport_probe",
        trial_id="probe_0",
        request_timeout_s=120,
        max_connect_attempts=1,
    ) as client:
        client.call("reset")
        started = time.perf_counter()
        client.call("update_obs", obs)
        actions = client.call("get_action")
        elapsed = time.perf_counter() - started
        packed = np.stack([
            np.concatenate([np.asarray(a[key]).reshape(-1) for key in (
                "left_ee_pose", "left_ee_joint_state", "right_ee_pose", "right_ee_joint_state"
            )])
            for a in actions
        ])
        assert packed.shape == (24, 16), packed.shape
        assert np.isfinite(packed).all()
        client.call("trial_end", {"kind": "synthetic_transport_probe"})
    np.save(args.output / "actions.npy", packed)
    report = {
        "kind": "synthetic_native_websocket_probe",
        "closed_loop": False,
        "url": args.url,
        "executed_action_shape": list(packed.shape),
        "finite": True,
        "request_seconds": elapsed,
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
