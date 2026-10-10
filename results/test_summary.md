## Test results (95% intervals from 2,000 field-level bootstrap resamples)

| Model | Level | Macro-F1 | Balanced acc. | OA (always-biggest-class baseline) | AUC |
|---|---|---|---|---|---|
| primary | field (n=412) | 0.560 [0.496, 0.625] | 0.554 [0.494, 0.620] | 0.701 (0.653) | 0.837 [0.800, 0.870] |
| s1_s2 | field (n=412) | 0.609 [0.545, 0.668] | 0.609 [0.543, 0.674] | 0.743 (0.653) | 0.882 [0.850, 0.910] |
| asc_only | field (n=412) | 0.536 [0.471, 0.597] | 0.537 [0.472, 0.602] | 0.667 (0.653) | 0.798 [0.760, 0.832] |
| s1+embedding | field (n=412) | 0.597 [0.529, 0.657] | 0.601 [0.533, 0.668] | 0.714 (0.653) | 0.860 [0.827, 0.889] |
| pixel model (map) | pixel (n=11653, 412 fields) | 0.517 [0.474, 0.558] | 0.528 [0.485, 0.571] | 0.647 (0.675) | 0.783 [0.750, 0.816] |
| pixel model, averaged per field | field | 0.555 | 0.543 | 0.704 | 0.824 |

### Primary (two-orbit Sentinel-1) minus each alternative, paired field bootstrap

| Alternative | Metric | Difference | 95% interval | Share of resamples where primary is better |
|---|---|---|---|---|
| s1_s2 | f1 | -0.050 | [-0.101, +0.000] | 0.03 |
| s1_s2 | auc | -0.046 | [-0.064, -0.028] | 0.00 |
| asc_only | f1 | +0.024 | [-0.035, +0.084] | 0.78 |
| asc_only | auc | +0.039 | [+0.018, +0.061] | 1.00 |
| s1+embedding | f1 | -0.037 | [-0.107, +0.028] | 0.14 |
| s1+embedding | auc | -0.024 | [-0.045, -0.003] | 0.01 |

### Per class (producer's accuracy = recall, user's accuracy = precision)

```
level        model       cls    PA    UA    F1   AUC    n
field      primary     Wheat 0.814 0.811 0.813 0.827  269
field      primary   Mustard 0.545 0.522 0.533 0.817  110
field      primary Sugarcane 0.303 0.370 0.333 0.865   33
field        s1_s2     Wheat 0.851 0.858 0.854 0.880  269
field        s1_s2   Mustard 0.582 0.566 0.574 0.845  110
field        s1_s2 Sugarcane 0.394 0.406 0.400 0.922   33
field     asc_only     Wheat 0.777 0.786 0.781 0.779  269
field     asc_only   Mustard 0.500 0.487 0.493 0.752  110
field     asc_only Sugarcane 0.333 0.333 0.333 0.863   33
field s1+embedding     Wheat 0.807 0.854 0.830 0.856  269
field s1+embedding   Mustard 0.573 0.504 0.536 0.810  110
field s1+embedding Sugarcane 0.424 0.424 0.424 0.915   33
pixel        pixel     Wheat 0.745 0.799 0.771 0.762 7864
pixel        pixel   Mustard 0.470 0.419 0.443 0.749 2820
pixel        pixel Sugarcane 0.370 0.309 0.337 0.838  969
```

### confusion_field_asc_only (rows = truth, columns = prediction; counts, then row share)

```
           Wheat  Mustard  Sugarcane
Wheat        209       47         13
Mustard       46       55          9
Sugarcane     11       11         11

           Wheat  Mustard  Sugarcane
Wheat       0.78     0.17       0.05
Mustard     0.42     0.50       0.08
Sugarcane   0.33     0.33       0.33
```

### confusion_field_primary (rows = truth, columns = prediction; counts, then row share)

```
           Wheat  Mustard  Sugarcane
Wheat        219       42          8
Mustard       41       60          9
Sugarcane     10       13         10

           Wheat  Mustard  Sugarcane
Wheat       0.81     0.16       0.03
Mustard     0.37     0.55       0.08
Sugarcane   0.30     0.39       0.30
```

### confusion_field_s1+embedding (rows = truth, columns = prediction; counts, then row share)

```
           Wheat  Mustard  Sugarcane
Wheat        217       45          7
Mustard       35       63         12
Sugarcane      2       17         14

           Wheat  Mustard  Sugarcane
Wheat       0.81     0.17       0.03
Mustard     0.32     0.57       0.11
Sugarcane   0.06     0.52       0.42
```

### confusion_field_s1_s2 (rows = truth, columns = prediction; counts, then row share)

```
           Wheat  Mustard  Sugarcane
Wheat        229       31          9
Mustard       36       64         10
Sugarcane      2       18         13

           Wheat  Mustard  Sugarcane
Wheat       0.85     0.12       0.03
Mustard     0.33     0.58       0.09
Sugarcane   0.06     0.55       0.39
```

### confusion_pixel_field_vote (rows = truth, columns = prediction; counts, then row share)

```
           Wheat  Mustard  Sugarcane
Wheat        223       40          6
Mustard       45       58          7
Sugarcane     11       13          9

           Wheat  Mustard  Sugarcane
Wheat       0.83     0.15       0.02
Mustard     0.41     0.53       0.06
Sugarcane   0.33     0.39       0.27
```

### confusion_pixel_pixel (rows = truth, columns = prediction; counts, then row share)

```
           Wheat  Mustard  Sugarcane
Wheat       5857     1511        496
Mustard     1187     1325        308
Sugarcane    286      324        359

           Wheat  Mustard  Sugarcane
Wheat       0.74     0.19       0.06
Mustard     0.42     0.47       0.11
Sugarcane   0.30     0.33       0.37
```