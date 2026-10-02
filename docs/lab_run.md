# RGB + LiDAR lab run (target: under 10 minutes at the robot)

One script, run on the control PC, deploys to the robot computer, checks
health and latency, records three short captures and stops. Everything that
needs thinking (camera calibration, LiDAR-to-camera alignment, latency,
acceptance) runs afterwards on the control PC from the capture, with the robot
switched off. Nothing in these scripts commands robot motion; the operator
moves the robot with the remote in the yaw step.

```bash
scripts/lab/run_lab.sh all chair:1.5:0 chair:3.0:0 chair:2.0:0.8
```

`label:x:y` are the taped object positions in `base_link` (x forward, y left,
metres) to the point on each object's surface facing the robot. Each step can
also be run alone (`run_lab.sh deploy|up|probe|checkerboard|scene ...|yaw|down|pull|offline`).

## Before the session (no robot)

* Print the 7x6-inner-corner, 25 mm checkerboard at 100 %, measure a square;
  if it is not 25.0 mm set `SQUARE_M` for the offline step.
* Tape marks for the objects; objects with flat faces toward the robot.
* Robot charged; remote in hand; cable or lab wifi to the robot computer.

## Timeline

| step | robot time | what happens | operator |
|---|---|---|---|
| deploy | ~2-3 min (first build), ~20 s after | robot-computer clock set from the PC, sources copied, `sysctl net.core.rmem_max`, GStreamer/Python/DDS checks, `colcon build` | nothing |
| up | ~20 s | base-stack LiDAR relay, C++ camera (Jetson hardware decoder, automatic fallback to software decode if it yields no frames), `lidar_depth_node` | nothing |
| probe | 15 s | PASS/FAIL table: camera fps, arrival-to-publish p95, DDS delivery, LiDAR and odometry rates, depth density | read the table |
| checkerboard | <= 2 min, stops early | records at 3 Hz, prints a 4x4 coverage map | sweep the board over corners and centre, 0.5-1.5 m, tilted |
| scene | 15 s | robot still, objects at the taped marks | board out of view |
| yaw | 30 s | full-rate frames + odometry | yaw slowly left/right with the remote (~0.5 rad/s) |
| down + pull | ~1-2 min | stop, copy ~150-250 MB to the PC | robot can be powered off |
| offline | ~1-2 min, no robot | intrinsics, latency, extrinsics, acceptance; `calibration/summary.txt` | read the summary |

## Pass criteria

Probe (live): camera >= 12 fps, arrival-to-publish p95 <= 25 ms, DDS delivery
>= 0.95, LiDAR >= 10 Hz, odometry >= 50 Hz, depth >= 3 Hz with >= 2000 valid
pixels. Offline: see `scripts/calib/README.md` (intrinsics RMS <= 0.5 px,
latency correlation >= 0.6, extrinsics alignment, every taped object within
0.15 m).

## Using the result

```bash
GSN_INTRINSICS=<run>/calibration/front_camera_intrinsics.yaml \
GSN_EXTRINSICS=<run>/calibration/front_camera_extrinsics.yaml \
GSN_LATENCY_S=<from latency.txt> \
  bash ~/gsn_lab/bin/jetson_lab.sh up      # on the robot computer, after copying the files there
```

or pass them to `deploy.launch.py` as `camera_intrinsics_file`,
`camera_extrinsics_file` (and `latency_s` on `go2_front_camera`). Until both
files exist, `lidar_depth_node` withholds depth outside these lab runs.

## If something fails

| symptom | likely cause | action |
|---|---|---|
| deploy: `FAIL DDS refused the 16MB receive buffer` | `rmem_max` not raised | rerun deploy; check sudo on the robot computer |
| deploy: `WARN DDS config does not name enP8p1s0` | DDS bound to the wrong NIC: topics list but carry no data | fix `CYCLONEDDS_URI` in the unitree setup before continuing |
| probe: camera fps 0 | multicast not reaching the NIC | `sudo tcpdump -i enP8p1s0 -c 5 udp port 1720` on the robot computer |
| probe: DDS delivery < 0.95 | receive buffers, CPU load | check `sysctl net.core.rmem_max`, `tegrastats` |
| probe: lidar/odom 0 | relay inputs missing | `ros2 topic hz /utlidar/robot_odom /utlidar/cloud_deskewed` |
| checkerboard: coverage stays low | board too far / one region | bring it closer, cover the empty cells of the map |
| offline latency: yaw-rate RMS < 0.2 | yawed too slowly | redo `run_lab.sh yaw` (30 s) |

## Rehearsal without the robot

`jetson_lab.sh` runs on a desktop with `GSN_REHEARSAL=1` (no sudo, loopback
camera, best-effort DDS buffer), `scripts/lab/fake_go2_sensors.py` (robot-clock
odometry and LiDAR) and `scripts/bench/rtp_counter_sender.py` (RTP H.264 camera
stream). The full robot-side sequence (prepare, up with decoder fallback,
probe, three captures, down) has been rehearsed this way; the offline step's
PASS path has only been exercised on synthetic captures.
