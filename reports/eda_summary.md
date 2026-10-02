# User 1 EDA summary

- Rows: 4,000; hits: 2,000; non-hits: 2,000.
- Mode B uses MSD-derived features; this is a methodology reproduction, not an exact Spotify-feature reproduction.
- `artist_score` uses Billboard events from 1986 onward that occur strictly before the song reference date.
- Chart rank, peak position, and weeks on chart are excluded from the model table.

## Numeric descriptive statistics by class

|                       |        0 |        1 |
|:----------------------|---------:|---------:|
| ('tempo', 'count')    | 2000     | 2000     |
| ('tempo', 'mean')     |  124.958 |  121.341 |
| ('tempo', 'std')      |   35.215 |   28.674 |
| ('tempo', 'min')      |    0     |    0     |
| ('tempo', '25%')      |   98.296 |   98.253 |
| ('tempo', '50%')      |  122.384 |  120.081 |
| ('tempo', '75%')      |  147.817 |  138.98  |
| ('tempo', 'max')      |  246.962 |  235.019 |
| ('loudness', 'count') | 2000     | 2000     |
| ('loudness', 'mean')  |   -9.24  |   -6.923 |
| ('loudness', 'std')   |    4.951 |    2.632 |
| ('loudness', 'min')   |  -40.035 |  -23.383 |
| ('loudness', '25%')   |  -11.51  |   -8.441 |
| ('loudness', '50%')   |   -7.962 |   -6.512 |
| ('loudness', '75%')   |   -5.774 |   -5.008 |
| ('loudness', 'max')   |    0.193 |   -0.726 |
| ('duration', 'count') | 2000     | 2000     |
| ('duration', 'mean')  |  248.864 |  253.802 |
| ('duration', 'std')   |  121.545 |   62.95  |
| ('duration', 'min')   |    5.955 |   13.296 |
| ('duration', '25%')   |  187.526 |  216.058 |
| ('duration', '50%')   |  231.575 |  241.854 |
| ('duration', '75%')   |  286.661 |  275.023 |
| ('duration', 'max')   | 2198.93  |  746.396 |

## Correlation notes

Correlations are descriptive only and must not be interpreted as causal effects.

|              |   tempo |   loudness |   duration |   artist_score |    hit |
|:-------------|--------:|-----------:|-----------:|---------------:|-------:|
| tempo        |   1     |      0.122 |     -0.039 |         -0.038 | -0.056 |
| loudness     |   0.122 |      1     |     -0.059 |          0.21  |  0.281 |
| duration     |  -0.039 |     -0.059 |      1     |          0.022 |  0.026 |
| artist_score |  -0.038 |      0.21  |      0.022 |          1     |  0.663 |
| hit          |  -0.056 |      0.281 |      0.026 |          0.663 |  1     |
