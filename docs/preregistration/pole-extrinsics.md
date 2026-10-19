# Pre-registration: camera extrinsics from the 2026-10-15 pole captures

Registered 2026-10-19, before the fit was run. The tool is
`scripts/calib/extrinsics_from_poles.py`. Thresholds below do not change after
the first run; a deviation is written into the results, not edited in here.

## Background

On 2026-10-15 the measured intrinsics passed (301 checkerboard views, RMS
0.430 px). The edge-based extrinsics tool (`extrinsics_from_capture.py`) then
exited 2 on all 15 scene captures, from the nominal start and from a start
corrected by a 2-D LiDAR fit, and `acceptance_from_capture.py` failed on S1-S5
(edge alignment 11.7-12.3 px against a 2.0 px limit; range within 0.15 m on 4
of 5). The scenes have few sharp edges and about 4 degrees of camera-LiDAR
pitch that the 2-D fit does not model. Eight single-pole captures (C1-C8; an
orange pole at tape-measured spots 4-12 ft away, 6 standing, 2 lying) exist and
were used only for the 2-D LiDAR fit. This registration uses them differently:
the pole is a compact, high-contrast target, so its pixels can be matched to
LiDAR returns directly without relying on image edges.

## Claim

A 6-DOF camera pose in the LiDAR base frame, fitted to the C1-C8 pole
captures with the measured intrinsics, projects LiDAR returns onto objects it
has never seen well enough to pass the repository's unchanged acceptance bar.

## Procedure

1. Pole pixels: orange HSV mask, largest connected component. Pole LiDAR points:
   the cluster within 0.6 m of the tape position (selector only; tape does not
   enter the objective), shifted to the median, 0.08 m radius, 0.10-1.2 m
   above the RANSAC floor plane. The C5 and C7 clusters are seeded away from a
   table leg, as in the 2-D fit.
2. Objective: mean distance (px, capped at 60) from each projected pole point
   to the mask, plus the squared pixel error between the mask's bottom-centre
   and the projected floor point under the pole axis.
3. Fit all six extrinsic parameters of the camera in the raw LiDAR base frame
   (Nelder-Mead from the nominal pose and from four perturbed starts; best
   objective wins).

## Evidence and bar

| Check | Data | PASS |
|---|---|---|
| Leave-one-pole-out | fit on 7 poles, score the 8th | mean pole-point-to-mask distance <= 4 px and bottom-point error <= 8 px on every held-out pole |
| Held-out acceptance | `acceptance_from_capture.py` on S1-S5 (never used in the fit), measured intrinsics, fitted extrinsics, defaults unchanged | every scene: object range within 0.15 m and mean silhouette-to-edge distance <= 2.0 px |
| Stability | start from five perturbed initial poses | all five end within 0.5 degree and 1 cm of each other |

## Kill criterion

If leave-one-pole-out or stability fails, the fitted file is not shipped as
`nominal: false` and the result is recorded as a failure. If both pass and the
S1-S5 acceptance fails, the file is kept as a candidate (still `nominal: true`
in the repository config), because passing on poles and failing on unseen
objects means the fit is not yet a calibration.

## Known limits, stated in advance

The scene reference positions for S1-S5 come from one operator's tape and may
be face or centre of the object (the 10-15 desk notes could not tell). The
range check inherits that uncertainty. The edge check does not. Both are
reported separately.

## Results (2026-10-19)

Verdict against the registration: **not met**. The fitted pose is kept as a candidate
(`front_camera_extrinsics_candidate_20261019.yaml`, `nominal: true`); the depth gate stays closed.

What happened, in order, including the deviations:

1. First run: the tape-window median shift locked C7 onto a table leg (60 px, no fit). Cluster selection was
   rewritten (connected 3 cm clusters, at most 0.35 m wide, nearest the tape position) before any held-out data
   was scored.
2. Second run (fixed selection): leave-one-out point-to-mask 0.2-1.0 px, bottom point 1.8-8.3 px (C1 8.25 px,
   0.25 px over the registered 8 px). Held-out S1-S5 acceptance with this pose: FAIL on all five (edge 11.3-12.1 px
   against 2.0, object range +0.18 to +0.23 m, S4 window artifact as on 10-15).
3. A synthetic test with a known pose then exposed three defects in my tool, none related to the data: the bottom
   term compared the mask's near-surface bottom with the pole axis (about 10 px bias at 1.7 m), the floor plane was
   fitted in the base frame and varied 1 degree between random seeds (now the odom-frame floor-height histogram),
   and Nelder-Mead stalled in a flat valley (now bilinear lookup plus a restart). The bottom term is now scaled in
   metres, because the LiDAR cluster centroid is only good to about 2 cm, which is 10-16 px at 1.7 m. This is a
   deviation from the registered "squared pixel error".
4. Final run (these defects fixed, same data): restarts agree to 0.02 deg and 1 mm. Leave-one-out point-to-mask
   0.36-0.93 px on every pole. Bottom point 1.2-13.3 px, which is 0.6-2.4 cm. The registered pixel bar (8 px)
   fails on C2 (10.6 px) and C3 (13.3 px); the metric bar (4 cm) used from then on passes.
   Held-out S1-S5 acceptance again fails identically (edge 11.3-12.1 px).

Two findings that bear on how to read the failure:

* The edge check has no power in these scenes. On S1 it reads 11.3 px for the fitted pose and 11.8-12.5 px for the
  same pose rotated by 5 degrees (yaw, pitch, roll) or 8 degrees (yaw). A check that scores a good and a 5-8
  degree wrong pose alike cannot reject or accept a calibration here.
* Post-hoc, not registered: on the orange cup stacks of S3 (1.1 m, lying) and S4 (2.5 m), which were never in the
  fit, LiDAR returns on the cup land 0.30 and 0.46 px from the cup mask with the fitted pose, against 22.8 and
  31.2 px with the nominal pose. S5 gave only 41 cluster returns and is inconclusive.

Decisive next test: capture a fresh set of poles and a chair, with a tape-measured chair position, on the robot
(no motion, ~10 minutes) and run `extrinsics_from_poles.py --evaluate`
(bar: every pole at most 2 px point-to-mask and 4 cm bottom point; chair range within 0.15 m of tape to the
surface facing the robot). A pass opens the gate by setting `nominal: false` on the file.
