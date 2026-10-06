# Streamlit deployment

The Streamlit entry point is `app/app.py`. It loads the selected random forest at
`models/best_pipeline.joblib` and verifies it against `models/metadata.json`.
The deployment version is `random_forest_v1_deployment`: 300 trees, maximum
depth 8, minimum leaf size 5, and four inputs. Its recorded artist-disjoint
test accuracy is 82.48%, with precision 92.64%, recall 70.25% and ROC-AUC 0.8767.
It was chosen for deployment after reviewing the original comparison;
this change does not create a fresh independent test result.

Live demo: https://hitpredict-demo.streamlit.app/

## Run locally

From the repository root:

```bash
python -m pip install -r app/requirements.txt
streamlit run app/app.py
```

## Deploy on Streamlit Community Cloud

1. Push the project, including the Git LFS model artifact, to GitHub.
2. At [share.streamlit.io](https://share.streamlit.io), create an app from
   `maddox095/ML-Mini-Project`, branch `main`, entry point `app/app.py`.
3. Choose a public subdomain, deploy, and wait for the build to finish.
4. Test two predictions that change at least two values, then capture the
   title, inputs, predicted class, probability, model evidence and URL.

Streamlit Community Cloud supports Git LFS repositories, so the deployed build
can retrieve the published `.joblib` model. No secrets are needed.
