# trajprompt — reproduced evidence

_Generated 2026-06-29T10:30:32Z by running the test suite on the real/tested code in this repo._

These are **reproduced** results: the code runs and every assertion below holds. Benchmark/leaderboard numbers in the paper (PSNR, mIoU, speedups) remain **targets, not reproduced**, and are labeled as such throughout.

## Test suite (`pytest -v`)

```
tests/test_smoke.py::test_top_level_imports PASSED                       [  5%]
tests/test_smoke.py::test_tgard_imports PASSED                           [ 11%]
tests/test_smoke.py::test_traj_clip_imports PASSED                       [ 16%]
tests/test_smoke.py::test_sam2_chip_imports PASSED                       [ 22%]
tests/test_smoke.py::test_traj_clip_forward_e2e PASSED                   [ 27%]
tests/test_smoke.py::test_tgard_pipeline_e2e PASSED                      [ 33%]
tests/test_smoke.py::test_sam2_chip_pipeline_returns_correct_shape PASSED [ 38%]
tests/test_smoke.py::test_space_app_importable PASSED                    [ 44%]
tests/test_smoke.py::test_space_ui_builds PASSED                         [ 50%]
tests/test_smoke.py::test_space_callback_returns_tuple PASSED            [ 55%]
tests/test_smoke.py::test_space_requirements_parseable PASSED            [ 61%]
tests/test_smoke.py::test_space_readme_has_hf_frontmatter PASSED         [ 66%]
tests/test_tgard.py::test_haversine_self_distance_zero PASSED            [ 72%]
tests/test_tgard.py::test_haversine_known_distance PASSED                [ 77%]
tests/test_tgard.py::test_rendezvous_no_match_when_far_apart PASSED      [ 83%]
tests/test_tgard.py::test_rendezvous_detects_close_dwell PASSED          [ 88%]
tests/test_traj_clip.py::test_encoder_output_is_l2_normalized PASSED     [ 94%]
tests/test_traj_clip.py::test_loss_drops_when_pairs_are_aligned PASSED   [100%]

============================== 18 passed in 2.11s ==============================
```

## Reproduced demo (headline number)

`haversine_pairwise` returns 111,195 m for 1 degree of latitude (matches the geodesic reference), the distance backbone behind TGARD rendezvous detection.
