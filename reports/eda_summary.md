# User 1 EDA summary

- Reproduction mode: **B** (MSD-native methodology reproduction).
- Rows: 198; hits: 99; non-hits: 99.
- Artist Score uses a strictly earlier Billboard event; see `artist_score_audit.csv`.
- Billboard outcome metadata is excluded from the model feature list.

## Feature availability

| feature   | source               |   missing_rows | model_input   |
|:----------|:---------------------|---------------:|:--------------|
| tempo     | Million Song Dataset |              0 | True          |
| loudness  | Million Song Dataset |              0 | True          |
| duration  | Million Song Dataset |              0 | True          |

## Numeric descriptive statistics

|                       |         0 |         1 |
|:----------------------|----------:|----------:|
| ('tempo', 'count')    |  99       |  99       |
| ('tempo', 'mean')     | 123.902   | 122.164   |
| ('tempo', 'std')      |  39.1228  |  29.4359  |
| ('tempo', 'min')      |  57.091   |  61.971   |
| ('tempo', '25%')      |  95.058   |  98.7265  |
| ('tempo', '50%')      | 119.562   | 118.273   |
| ('tempo', '75%')      | 140.197   | 140.284   |
| ('tempo', 'max')      | 241.877   | 206.02    |
| ('loudness', 'count') |  99       |  99       |
| ('loudness', 'mean')  |  -8.50636 |  -6.61665 |
| ('loudness', 'std')   |   3.87399 |   2.26092 |
| ('loudness', 'min')   | -23.996   | -12.806   |
| ('loudness', '25%')   | -11.206   |  -7.779   |
| ('loudness', '50%')   |  -7.8     |  -6.295   |
| ('loudness', '75%')   |  -5.3995  |  -5.2055  |
| ('loudness', 'max')   |  -1.869   |  -1.81    |
| ('duration', 'count') |  99       |  99       |
| ('duration', 'mean')  | 251.299   | 268.067   |
| ('duration', 'std')   |  81.2914  |  77.797   |
| ('duration', 'min')   |  43.3628  | 160.313   |
| ('duration', '25%')   | 206.511   | 224.574   |
| ('duration', '50%')   | 239.882   | 249.391   |
| ('duration', '75%')   | 298.174   | 288.039   |
| ('duration', 'max')   | 540.473   | 652.408   |

## Correlation notes

Correlations are descriptive only. No Billboard rank, peak, or weeks-on-chart field is a feature.

|              |   tempo |   loudness |   duration |   artist_score |    hit |
|:-------------|--------:|-----------:|-----------:|---------------:|-------:|
| tempo        |   1     |      0.141 |     -0.085 |          0.068 | -0.025 |
| loudness     |   0.141 |      1     |      0.046 |          0.357 |  0.287 |
| duration     |  -0.085 |      0.046 |      1     |          0.15  |  0.105 |
| artist_score |   0.068 |      0.357 |      0.15  |          1     |  0.638 |
| hit          |  -0.025 |      0.287 |      0.105 |          0.638 |  1     |
