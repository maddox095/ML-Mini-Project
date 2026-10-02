# User 1 EDA summary

- Rows: 4,000; hits: 2,000; non-hits: 2,000.
- Mode B uses MSD-derived features; this is a methodology reproduction, not an exact Spotify-feature reproduction.
- `artist_score` uses Billboard events from 1986 onward that occur strictly before the song reference date.
- Chart rank, peak position, and weeks on chart are excluded from the model table.

## Numeric descriptive statistics by class

|                       |        0 |        1 |
|:----------------------|---------:|---------:|
| ('tempo', 'count')    | 2000     | 2000     |
| ('tempo', 'mean')     |  125.424 |  121.566 |
| ('tempo', 'std')      |   36.025 |   28.378 |
| ('tempo', 'min')      |   10.456 |   41.158 |
| ('tempo', '25%')      |   97.5   |   98.253 |
| ('tempo', '50%')      |  121.979 |  120.081 |
| ('tempo', '75%')      |  147.922 |  139.161 |
| ('tempo', 'max')      |  248.64  |  235.019 |
| ('loudness', 'count') | 2000     | 2000     |
| ('loudness', 'mean')  |   -9.354 |   -6.936 |
| ('loudness', 'std')   |    4.836 |    2.627 |
| ('loudness', 'min')   |  -48.671 |  -21.077 |
| ('loudness', '25%')   |  -11.71  |   -8.517 |
| ('loudness', '50%')   |   -8.233 |   -6.508 |
| ('loudness', '75%')   |   -5.82  |   -5.022 |
| ('loudness', 'max')   |   -0.678 |   -0.726 |
| ('duration', 'count') | 2000     | 2000     |
| ('duration', 'mean')  |  246.723 |  254.123 |
| ('duration', 'std')   |  120.462 |   62.97  |
| ('duration', 'min')   |   13.009 |   14.471 |
| ('duration', '25%')   |  184.999 |  216.306 |
| ('duration', '50%')   |  229.877 |  242.037 |
| ('duration', '75%')   |  284.022 |  275.089 |
| ('duration', 'max')   | 2513.16  |  746.396 |

## Correlation notes

Correlations are descriptive only and must not be interpreted as causal effects.

|              |   tempo |   loudness |   duration |   artist_score |    hit |
|:-------------|--------:|-----------:|-----------:|---------------:|-------:|
| tempo        |   1     |      0.119 |     -0.026 |         -0.049 | -0.059 |
| loudness     |   0.119 |      1     |     -0.02  |          0.221 |  0.297 |
| duration     |  -0.026 |     -0.02  |      1     |          0.02  |  0.038 |
| artist_score |  -0.049 |      0.221 |      0.02  |          1     |  0.662 |
| hit          |  -0.059 |      0.297 |      0.038 |          0.662 |  1     |
