"""Fast sensor_msgs/Image filling.

Assigning ``bytes`` to the ``uint8[]`` data field makes rclpy validate every
element in Python (~96 ms for a 720p BGR frame); an ``array.array('B')`` is
taken as a buffer (~0.1 ms).
"""

from __future__ import annotations

import array

import numpy as np
from sensor_msgs.msg import Image

_ENCODINGS = {
    (np.dtype(np.uint8), 3): "bgr8",
    (np.dtype(np.uint8), 1): "mono8",
    (np.dtype(np.uint16), 1): "16UC1",
}


def fill_image(msg: Image, img: np.ndarray, encoding: str | None = None) -> Image:
    channels = 1 if img.ndim == 2 else img.shape[2]
    enc = encoding or _ENCODINGS.get((img.dtype, channels))
    if enc is None:
        raise ValueError(f"no default encoding for dtype {img.dtype} with {channels} channels")
    img = np.ascontiguousarray(img)
    msg.height, msg.width = int(img.shape[0]), int(img.shape[1])
    msg.encoding = enc
    msg.is_bigendian = 0
    msg.step = int(img.strides[0])
    msg.data = array.array("B", img.tobytes())
    return msg


def image_to_numpy(msg: Image) -> np.ndarray:
    """Decode bgr8/rgb8/mono8/16UC1 without cv_bridge; returns BGR for colour."""
    if msg.encoding in ("bgr8", "rgb8"):
        img = np.frombuffer(bytes(msg.data), np.uint8).reshape(msg.height, msg.step)[:, : msg.width * 3]
        img = img.reshape(msg.height, msg.width, 3)
        return img[:, :, ::-1].copy() if msg.encoding == "rgb8" else img.copy()
    if msg.encoding == "mono8":
        return np.frombuffer(bytes(msg.data), np.uint8).reshape(msg.height, msg.step)[:, : msg.width].copy()
    if msg.encoding in ("16UC1", "mono16"):
        row = np.frombuffer(bytes(msg.data), np.uint16).reshape(msg.height, msg.step // 2)
        return row[:, : msg.width].copy()
    raise ValueError(f"unsupported encoding {msg.encoding!r}")
