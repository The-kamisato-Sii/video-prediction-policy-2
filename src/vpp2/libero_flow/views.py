"""Flow label views are independent of the fixed two-camera RGB conditions."""
RGB_CAMERAS = ("image", "wrist_image")
DEFAULT_MOTION_CAMERAS = ("image",)


def motion_cameras(value):
    if isinstance(value, str):
        raise ValueError("flow.motion_cameras must be a list, e.g. [image]")
    cameras = tuple(value)
    if cameras not in (("image",), RGB_CAMERAS):
        raise ValueError("flow.motion_cameras must be [image] or [image,wrist_image] in that order")
    return cameras


def check_checkpoint_views(payload, expected):
    # Historical v1 checkpoints were unconditionally dual-view.
    saved = tuple(payload.get("motion_cameras", RGB_CAMERAS))
    if saved != tuple(expected):
        raise ValueError(f"Checkpoint motion cameras {saved} differ from model {tuple(expected)}; start a new run")
