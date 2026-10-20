#!/usr/bin/env python3
"""
sim_scene_graph.py: perception stand-in for closed-loop simulation.

Publishes /semantic/scene_graph from the labeled objects in a go2_sim world
file, so the grounding -> navigation -> safety -> actuation chain can be run
end to end without a camera. It is NOT a perception result and must never be
quoted as one; RGB-D perception quality is measured separately on recorded
bags (eval/run_eval.py --mode rosbag).

Embeddings. The grounding scorer compares a CLIP text embedding of the query
with each object's embedding. Real image-to-text cosine similarity for a
correct match is ~0.25-0.32; text-to-text similarity between two different
labels is often 0.7-0.9. Using raw label text embeddings would therefore let
absent queries pass the clip floor. Each object's embedding is instead

    normalize(a * text("a photo of a <label>") + sqrt(1 - a^2) * r)

with r a fixed random unit vector orthogonal to the text embedding and
a = --clip-alignment (default 0.30), which reproduces the real similarity
scale for the correct label and scales down every other label accordingly.

World frame. go2_sim publishes odometry in its world frame (odom == world,
the robot starts at the world's start_pose), and slam_toolbox initialises
map == odom (verified in closed-loop sim: map->odom translation 0, 0 in x/y).
Objects are therefore published in the map frame with world coordinates.
SLAM drift is not modeled here; the evaluation measures arrival against
ground truth, so drift shows up as arrival error.
"""
from __future__ import annotations

import argparse
import math
import sys
import time

import numpy as np
import yaml


def _embeddings(labels: list[str], encoder_name: str, device: str, alignment: float,
                seed: int) -> dict:
    from go2_open_vocab_detector.backends import make_encoder

    enc = make_encoder(encoder_name)
    enc.load(device=device)
    rng = np.random.default_rng(seed)
    out = {}
    for label in sorted(set(labels)):
        t = np.asarray(enc.encode_text([f"a photo of a {label}"])[0], dtype=np.float64)
        t /= np.linalg.norm(t)
        r = rng.normal(size=t.shape)
        r -= (r @ t) * t
        r /= np.linalg.norm(r)
        e = alignment * t + math.sqrt(max(0.0, 1.0 - alignment ** 2)) * r
        out[label] = (e / np.linalg.norm(e)).astype(np.float32)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("world_file")
    ap.add_argument("--encoder", default="openclip_vit_b16")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--clip-alignment", type=float, default=0.30)
    ap.add_argument("--rate", type=float, default=2.0)
    ap.add_argument("--frame", default="map")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args(argv)

    world = yaml.safe_load(open(args.world_file))
    objects = world.get("objects", [])
    if not objects:
        print("world has no labeled objects", file=sys.stderr)
        return 2
    embeddings = _embeddings(
        [o["label"] for o in objects], args.encoder, args.device, args.clip_alignment, args.seed
    )

    import rclpy
    from geometry_msgs.msg import PoseStamped, Vector3
    from go2_semantic_msgs.msg import SceneGraph, SemanticObject
    from rclpy.qos import QoSProfile, ReliabilityPolicy

    rclpy.init()
    node = rclpy.create_node("sim_scene_graph")
    qos = QoSProfile(depth=10)
    qos.reliability = ReliabilityPolicy.RELIABLE
    pub = node.create_publisher(SceneGraph, "/semantic/scene_graph", qos)

    def build() -> SceneGraph:
        graph = SceneGraph()
        graph.header.stamp = node.get_clock().now().to_msg()
        graph.header.frame_id = args.frame
        for i, o in enumerate(objects):
            obj = SemanticObject()
            obj.object_id = f"sim-{i}-{o['label']}"
            obj.label = o["label"]
            obj.confidence = 0.9
            obj.pose = PoseStamped()
            obj.pose.header = graph.header
            obj.pose.pose.position.x = float(o["x"])
            obj.pose.pose.position.y = float(o["y"])
            obj.pose.pose.position.z = float(o.get("z", 0.4))
            obj.pose.pose.orientation.w = 1.0
            obj.dimensions_xyz = Vector3(
                x=float(o.get("size_x", 0.5)), y=float(o.get("size_y", 0.5)), z=float(o.get("size_z", 0.8))
            )
            emb = embeddings[o["label"]]
            obj.embedding = emb.tolist()
            obj.embedding_dim = int(emb.shape[0])
            obj.observation_count = 10
            graph.nodes.append(obj)
        graph.total_objects = len(graph.nodes)
        return graph

    node.get_logger().info(f"publishing {len(objects)} sim objects on /semantic/scene_graph")
    period = 1.0 / max(0.1, args.rate)
    try:
        while rclpy.ok():
            pub.publish(build())
            rclpy.spin_once(node, timeout_sec=0.0)
            time.sleep(period)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
