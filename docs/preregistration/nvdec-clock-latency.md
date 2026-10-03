# Pre-registration: NVDEC clock and front-camera latency

Registered 2026-10-02, before any A/B data exists. The procedure is
`run_lab.sh clockab` (robot side: `jetson_lab.sh clockab`) and the analysis is
`scripts/lab/clockab_analyze.py`, committed with this file. Neither changes
after the run: a deviation from anything below is written into the results,
not edited in here.

## Background

The C++ camera node's arrival->publish latency starts when the UDP packets of a
frame arrive (udpsrc running time, carried through the depayloader to the
decoded frame) and ends at its ROS publish. It therefore includes depayload,
parse, decode, colour conversion and publish. On the robot computer
(2026-10-02, no motion; see `docs/lab_run.md`) the node's p50 was about 18 ms
with two image readers and the per-window p95 was 18.6-25.4 ms, marginal
against the 25 ms probe limit. Subtracting the conversions (~6 ms) and the copy
plus DDS publish (1-2 ms per reader) left roughly 10 ms for decode, the largest
stage. Under the default `tegra_wmark` devfreq governor the NVDEC clock read
115 MHz, its floor, in 18 of 20 samples taken while decoding, against a maximum
of 858 MHz. `enable-max-performance=true` on nvv4l2decoder does not raise it.

One attempt to A/B the clock stopped on a script error before measuring
anything, so no A/B data has been seen.

## Claim

On the GO2's robot computer, decoding the front camera's 1280x720 H.264 stream
with nvv4l2decoder in `go2_front_camera_cpp`, holding the NVDEC clock at its
maximum lowers the per-window arrival->publish p50 by at least 2.0 ms relative
to the default governor (one session, robot still, static view).

## Arms

| Arm | NVDEC clock | VIC clock | Role |
|---|---|---|---|
| A | default | default | control |
| B | max | default | treatment, primary comparison B - A |
| C | max | max | adds VIC (used by nvvidconv); secondary, exploratory |

"max" is the `performance` governor when both engines offer it, otherwise
`min_freq` raised to `max_freq`; the method used is recorded. "default" is the
governor and `min_freq` read before the run, restored when it ends (also on
error or interrupt).

- No sham-trigger arm: the intervention is a fixed setting and nothing decides
  when to apply it. The switching procedure is equalised instead: before every
  phase, A included, both engines' settings are written and 5 s pass before
  measuring.
- No oracle arm: nothing in the treatment is a prediction.
- Positive control (manipulation check): each engine's governor and `cur_freq`
  are read with every 1 s latency window. Without this, a null could not be
  told apart from a clock write that silently did nothing.

## Design

- 6 blocks. Each block runs A, B and C once, in this order: 1 ABC, 2 BCA,
  3 CAB, 4 ACB, 5 CBA, 6 BAC. Every arm appears twice in every position and
  every ordered pair of neighbours appears twice, which balances drift and
  first-order carryover.
- A phase: write the settings, discard windows for 5 s (settle), then the next
  25 one-second windows are the measurement. The count is fixed, so every
  phase has the same exposure.
- Configuration for the whole run: `run_lab.sh up nvv4l2` only (relay, camera,
  depth); slam and semantic not running; robot still; static view with nobody
  walking through it; no nvpmodel or jetson_clocks change. Recorded at the
  start: original governors and `min_freq`, the method, the nvpmodel mode, the
  ROS node list and the deployed commit.

## Validity rules (fixed before seeing any latency)

- A phase is valid when it has 25 measured windows, every window has a p50 and
  fps >= 12, the camera's restart counter does not change, and:
  - A: both governors read their recorded defaults;
  - B: NVDEC `cur_freq` equals its `max_freq` in at least 90 % of windows;
  - C: the same for NVDEC and for VIC.
- A block counts for the primary comparison when its A and B phases are valid,
  and for comparisons with C when all three are.
- The run is INVALID, not a null, and is repeated when:
  - fewer than 5 of the 6 blocks count for the primary comparison, or
  - there is no clock contrast: the mean NVDEC `cur_freq` over the valid A
    phases is above 50 % of `max_freq`. The default governor would then already
    run NVDEC fast at this load, the premise of the test is false, and that is
    what gets reported.

## Primary analysis

- Per block i: d_i = median of the B windows' p50 minus median of the A
  windows' p50, in ms.
- Estimate: the mean of d_i over the counted blocks.
- 95 % CI: percentile bootstrap over blocks, 10,000 resamples,
  `numpy.random.default_rng(20261002)`.
- Exact two-sided sign-flip permutation p over all 2^n sign patterns of d_i
  (n = 6: smallest attainable p is 0.031).

## Decision and kill criterion

The first matching rule decides:

1. SUPPORTED: the CI upper bound is below 0 and the estimate is -2.0 ms or
   lower.
2. Otherwise the claim is dead and is published as a negative result, with the
   estimate and CI, in `RESULTS.md` and `docs/lab_run.md`, described as:
   - "detectable but below 2.0 ms" when the CI upper bound is below 0;
   - "slower with the clock held at max" when the CI lower bound is above 0;
   - "no meaningful effect" when the CI lies inside (-2.0, +2.0) ms;
   - "inconclusive" otherwise.
- Stopping rule: exactly 6 blocks. No early stop and no added blocks after
  seeing data.

## Secondary outcomes (exploratory, never a headline)

- B - A on the per-phase median of window p95; C - B on p50 and on p95.
  Sign-flip p-values for these three are Holm-adjusted together.
- Power: mean VDD_IN per phase from tegrastats, B - A and C - A; maximum tj
  temperature per phase, as a check on thermal drift.
- If the claim is SUPPORTED, whether to hold the clock in the deployed launch
  is a separate decision that weighs the power cost.

## Confounds

| Check | Answer |
|---|---|
| Onset | No alarms or detectors. The transient after a clock write is covered by the fixed 5 s settle. |
| Exposure | Exactly 25 windows per phase in every arm. |
| Replication | 6 paired blocks in one session on one robot. The claim is scoped to that; no claim about other days or units. |
| Bundling | The arms differ only in the engines' clock settings. The write procedure and tegrastats run identically in every phase. |
| Positive control | The per-window clock reading (manipulation check), plus the INVALID rule when A shows no contrast. |
| Outcome | The latency itself is the quantity of interest. No task-level claim is made. |
| Ceiling and floor | With decode at zero the conversions and publish (~7-8 ms) remain against a p50 of ~18 ms, so a 2 ms effect is not floor-limited. |
| Selection | Threshold, window count, block count, order and analysis are fixed here before any A/B data. The 115 MHz observation that motivates the test is not part of the A/B data. |
| Dose | Equal by construction: each arm holds its setting for the same 25 measured windows. |
| Leakage | Nothing is fitted or tuned on the data. |

## Post-run audit

Recompute every reported number from the pulled `windows.jsonl` with
`clockab_analyze.py` on a fresh clone, and list every deviation from this file
(including blocks or phases marked invalid and why) next to the result.
