"""Artist-group learning curves with fixed, declared course-model settings.

Only original training rows are loaded. Curves are development diagnostics;
parameters are declared here, with no tuning or threshold fitting on curve
validation labels. They do not constitute an independent final test.
"""
import numpy as np
import pandas as pd
from sklearn.base import clone
from threadpoolctl import threadpool_limits

from src.data.common import load_config, repo_path, sha256
from src.models.course_models import json_parameters, make_course_pipeline
from src.models.data import load_partition
from src.models.artifacts import write_json
from src.models.validation import make_folds

SETTINGS = {
    'logistic_regression': {'model__C': 1.0},
    'decision_tree': {'model__max_depth': 4, 'model__min_samples_leaf': 20},
    'random_forest': {'model__max_depth': 4, 'model__min_samples_leaf': 5,
                      'model__max_features': 'sqrt', 'model__max_samples': 1.0},
}


def run():
    destination = repo_path('reports/runs/course_learning_curve_v2')
    if destination.exists():
        raise FileExistsError('Preserve earlier learning curves')
    destination.mkdir(parents=True)
    config = load_config('configs/accuracy_v2.yaml')
    X, y, metadata = load_partition(config['protocol'], 'train')
    folds = make_folds(y, metadata, config['protocol'], count=5, seed=config['seed'])
    parameters = {}
    rows = []
    with threadpool_limits(limits=1):
        for name, settings in SETTINGS.items():
            base = make_course_pipeline(name, config).set_params(**settings)
            parameters[name] = json_parameters(base)
            for number, (training, validation) in enumerate(folds, 1):
                groups = np.sort(metadata.iloc[training].artist_key.unique())
                shuffled = np.random.default_rng(config['seed'] + number).permutation(groups)
                for fraction in [.25, .5, .75, 1.0]:
                    selected_groups = shuffled[:max(2, int(len(groups)*fraction))]
                    subset = training[metadata.iloc[training].artist_key.isin(selected_groups).to_numpy()]
                    assert not set(metadata.iloc[subset].artist_key) & set(metadata.iloc[validation].artist_key)
                    estimator = clone(base).fit(X.iloc[subset], y.iloc[subset])
                    rows.append({'model': name, 'fold': number, 'artist_fraction': fraction,
                        'training_rows': len(subset), 'training_artists': len(selected_groups),
                        'validation_rows': len(validation),
                        'train_accuracy': estimator.score(X.iloc[subset], y.iloc[subset]),
                        'validation_accuracy': estimator.score(X.iloc[validation], y.iloc[validation])})
            print('Completed fixed-setting learning curve:', name, flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(destination / 'fold_results.csv', index=False)
    summary = frame.groupby(['model', 'artist_fraction'], as_index=False).agg(
        training_rows_mean=('training_rows', 'mean'),
        train_accuracy_mean=('train_accuracy', 'mean'),
        cv_accuracy_mean=('validation_accuracy', 'mean'),
        cv_accuracy_std=('validation_accuracy', 'std'))
    summary.to_csv(destination / 'summary.csv', index=False)
    write_json(destination / 'manifest.json', {'status': 'complete', 'config': config,
        'settings': SETTINGS, 'full_pipeline_parameters': parameters,
        'test_sets_evaluated': False, 'threshold': .5, 'device': 'cpu',
        'source_sha256': sha256(repo_path('data/interim/model_table.parquet')),
        'code': repo_path('src/models/course_learning_curve.py').read_text(encoding='utf-8'),
        'interpretation': 'Fixed-setting training-only diagnostic; no guarantee from adding more data.'})
    return summary


if __name__ == '__main__':
    print(run().to_string(index=False))
