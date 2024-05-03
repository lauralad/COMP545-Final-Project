# COMP545 Final Project

## Access Deployed App
Click the link [Streamlit website](https://comp545-final-project.streamlit.app/?recording=adayjrv)

## Local Setup
```
git clone
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Candidates
In order to run the app, there is a zip contained with wl_data called "candidates" that needs to be unzipped in order for the app to run.
Unzip it so it is in the same place it currently is, so you should have wl_data/candidates/valid.jsonl.
The reason for doing this is that the original file was too large to be kept on git.

## Experiment
Open experiment.ipynb, and you can run the code if you like to regenerate the models' predictions
If you scroll to the bottom however, you can run the accuracy retrieval on the sample data, which is present within the experiment_data dir.