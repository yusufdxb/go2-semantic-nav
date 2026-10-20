"""CycloneDDS settings every deploy node needs, appended to the operator's URI.

Measured on the GO2 Jetson (2026-10-02): with two or more readers of the raw
720p image, CycloneDDS sent the frames as multicast DATA on the robot NIC
(~41 MB/s onto the GO2's internal network) and the camera's synchronous
publish went from p50 16 ms to 53 ms. Multicast stays on for discovery only,
so local readers get unicast over loopback.

The receive buffer asks for 16 MB (a raw frame is ~2.7 MB of UDP fragments)
but accepts less, so the stack still starts where net.core.rmem_max was not
raised; run scripts/lab/run_lab.sh deploy to raise it.
"""

from __future__ import annotations

DEPLOY_FRAGMENTS = (
    "<CycloneDDS><Domain><General><AllowMulticast>spdp</AllowMulticast></General></Domain></CycloneDDS>",
    '<CycloneDDS><Domain><Internal><SocketReceiveBufferSize min="default" max="16MB"/></Internal></Domain></CycloneDDS>',
)


def cyclonedds_uri(existing: str | None) -> str:
    """The operator's CYCLONEDDS_URI (interface binding etc.) followed by DEPLOY_FRAGMENTS."""
    parts = [existing.strip()] if existing and existing.strip() else []
    parts += [f for f in DEPLOY_FRAGMENTS if f not in (existing or "")]
    return ",".join(parts)
