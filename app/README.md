# Streamlit deployment

The Streamlit entry point is `app/app.py`. It loads the frozen v1 pipeline at
`models/best_pipeline.joblib` and verifies it against `models/metadata.json`.

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
