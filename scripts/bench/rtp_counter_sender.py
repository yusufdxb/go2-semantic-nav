#!/usr/bin/env python3
"""RTP H.264 sender whose frames carry their own frame counter, for latency benchmarks.

Each 1280x720 frame has a moving pattern (realistic encoder load) and a 24-bit
counter drawn twice as 48x48 black/white blocks in the top rows. The wall-clock
time each frame is handed to the encoder is written to --log (counter,send_ns).
Sends unicast to 127.0.0.1 by default: nothing leaves the machine.
"""
import argparse
import time

import gi
import numpy as np

gi.require_version("Gst", "1.0")
from gi.repository import Gst  # noqa: E402

BITS, BLOCK, PITCH, X0, ROWS = 24, 48, 52, 16, (0, 60)


def draw_counter(frame: np.ndarray, value: int) -> None:
    for y0 in ROWS:
        for i in range(BITS):
            v = 255 if (value >> i) & 1 else 0
            frame[y0:y0 + BLOCK, X0 + i * PITCH:X0 + i * PITCH + BLOCK] = v


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=56200)
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--log", required=True)
    ap.add_argument("--noise", action="store_true", help="incompressible frames (worst-case keyframe bursts)")
    a = ap.parse_args()
    Gst.init(None)
    pipe = Gst.parse_launch(
        f"appsrc name=src is-live=true format=time do-timestamp=true "
        f"caps=video/x-raw,format=BGR,width={a.width},height={a.height},framerate={a.fps}/1 "
        f"! videoconvert ! video/x-raw,format=I420 ! openh264enc ! video/x-h264,profile=constrained-baseline "
        f"! rtph264pay config-interval=1 pt=96 mtu=1400 ! udpsink host={a.host} port={a.port} sync=false"
    )
    src = pipe.get_by_name("src")
    pipe.set_state(Gst.State.PLAYING)
    rng = np.random.default_rng(0)
    if a.noise:
        base = rng.integers(40, 200, size=(a.height, a.width, 3), dtype=np.uint8)
    else:  # camera-like: smooth gradients, edges, mild texture
        yy, xx = np.mgrid[0:a.height, 0:a.width]
        base = np.stack([(xx * 255 // a.width), (yy * 255 // a.height), ((xx + yy) % 256)], axis=2).astype(np.uint8)
        for k in range(12):
            x0, y0 = rng.integers(0, a.width - 200), rng.integers(120, a.height - 150)
            base[y0:y0 + 140, x0:x0 + 180] = rng.integers(0, 255, size=3, dtype=np.uint8)
        base = np.clip(base.astype(np.int16) + rng.integers(-6, 7, size=base.shape), 0, 255).astype(np.uint8)
    period = 1.0 / a.fps
    t0 = time.monotonic()
    with open(a.log, "w") as log:
        log.write("counter,send_ns\n")
        n = 0
        while time.monotonic() - t0 < a.seconds:
            frame = np.roll(base, shift=n * 8, axis=1)
            draw_counter(frame, n)
            buf = Gst.Buffer.new_wrapped(frame.tobytes())
            log.write(f"{n},{time.time_ns()}\n")
            src.emit("push-buffer", buf)
            n += 1
            time.sleep(max(0.0, t0 + n * period - time.monotonic()))
    src.emit("end-of-stream")
    time.sleep(0.5)
    pipe.set_state(Gst.State.NULL)


if __name__ == "__main__":
    main()
