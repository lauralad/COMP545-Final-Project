# COMP545 Final Project

## Access Deployed App
! Check out Turn #10 with other options being default, it is especially clear in this turn that it actually performs the actions ! 

Click the link [Streamlit website](https://comp545-final-project.streamlit.app/?recording=adayjrv).

Given the unique setup of the app, choosing a turn means that Playwright will perform previous web browser actions in the background in order to reach the point of the specified turn, and it will execute the predicted action along with a screenshot proof of it (i.e. the last screenshot in column 2 is the predicted action performed)

## Local Setup
```
git clone
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Experiment
Open experiment.ipynb, and you can run the code if you like to regenerate the models' predictions
If you scroll to the bottom however, you can run the accuracy retrieval on the sample data, which is present within the experiment_data dir.
