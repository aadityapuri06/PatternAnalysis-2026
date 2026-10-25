## Test prostate Dice (mean over the 28 test volumes, per run)

| Model | Runs | Mean ± std over runs |
|---|---|---|
| 2D U-Net (baseline) | 0.748, 0.682, 0.739 | 0.723 ± 0.036 (n=3) |
| 2D Improved U-Net | 0.702, 0.664, 0.714 | 0.693 ± 0.026 (n=3) |
| 3D Improved U-Net (matched settings) | 0.768, 0.737, 0.797 | 0.767 ± 0.030 (n=3) |
| 3D Improved U-Net (lr 5e-4, 50 epochs) | 0.761 | 0.761 (1 run) |
| 2D Improved U-Net (lr 5e-4) | 0.699 | 0.699 (1 run) |

## Prostate Dice per test patient (mean over runs)

| Model | B006 | G021 | K018 | S035 | T009 | W029 |
|---|---|---|---|---|---|---|
| 2D U-Net (baseline) | 0.795 | 0.697 | 0.779 | 0.807 | 0.903 | 0.643 |
| 2D Improved U-Net | 0.725 | 0.671 | 0.718 | 0.801 | 0.887 | 0.644 |
| 3D Improved U-Net (matched settings) | 0.862 | 0.794 | 0.797 | 0.776 | 0.918 | 0.667 |
| 3D Improved U-Net (lr 5e-4, 50 epochs) | 0.864 | 0.800 | 0.803 | 0.756 | 0.915 | 0.636 |
| 2D Improved U-Net (lr 5e-4) | 0.752 | 0.722 | 0.699 | 0.803 | 0.913 | 0.623 |

3D (matched) beats the 2D baseline on 5 of 6 test patients.

## Dice by prostate third, rectum/bladder leakage and inference time (from predict.py --analyse)

| Model | Base | Middle | Apex | Leak rectum % | Leak bladder % | s / volume |
|---|---|---|---|---|---|---|
| 2D U-Net (baseline) | 0.621 (1 run) | 0.818 (1 run) | 0.818 (1 run) | 0.40 | 2.67 | 0.19 |
| 2D Improved U-Net | 0.584 (1 run) | 0.763 (1 run) | 0.788 (1 run) | 0.74 | 1.56 | 0.21 |
| 3D Improved U-Net (matched settings) | 0.685 (1 run) | 0.830 (1 run) | 0.797 (1 run) | 0.47 | 0.15 | 0.15 |
| 3D Improved U-Net (lr 5e-4, 50 epochs) | 0.698 (1 run) | 0.808 (1 run) | 0.796 (1 run) | 0.24 | 0.81 | 0.16 |
| 2D Improved U-Net (lr 5e-4) | n/a | n/a | n/a | n/a | n/a | n/a |
