"""Editorial-print figure theme shared across projects.

Source of truth: physx-newton-bench. One line to adopt:

    import physx_style; physx_style.apply()

Then plot as usual; every figure inherits the cream canvas, muted frame,
and the physx-blue / newton-green palette. `PALETTE` and `COLORS` are
exposed for scripts that need an explicit per-series color.
"""
import os
import matplotlib.pyplot as plt

_STYLE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "physx_style.mplstyle")

# Editorial-print ink used for annotations/titles when set explicitly.
INK = "#0b0b0b"
CREAM = "#fcfcfb"
PALETTE = ["#2a78d6", "#1baf7a", "#d08a2e", "#b0472b", "#6a5acd", "#52514e"]
# Canonical two-backend mapping (physx-newton-bench).
COLORS = {"physx": "#2a78d6", "newton": "#1baf7a"}


def apply():
    """Activate the shared style. Safe to call more than once."""
    plt.style.use(_STYLE)


def cmap_cycle(n):
    """n categorical colors from the editorial palette (drop-in for plt.cm.tabX).

    Returns a list of hex strings, wrapping the palette if n exceeds its length.
    """
    return [PALETTE[i % len(PALETTE)] for i in range(int(n))]
