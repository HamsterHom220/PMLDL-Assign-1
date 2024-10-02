<h1>App description</h1>
Overall, this is a simple web app for predicting what the given text review is about
- among six categories defined by the train dataset: 

    {   'toys games': 0,
        'health personal care': 1,
        'beauty': 2,
        'baby products': 3,
        'pet supplies': 4,
        'grocery gourmet food': 5
    }

The user enters their review in a text field, presses the 'submit' button,
and frontend sends the GET/predict request to the API, which runs the inference
of a pretrained model. Then the response is fetched to the same page, and
the resulting category is displayed as a string.

<h1>Stage 1. Data engineering</h1>
<h2>Input:</h2>
<ul>- Raw data</ul>
<h2>Output:</h2>
<ul>- Processed data</ul>

Data loading, cleaning and splitting are performed by utility functions
defined in /code/datasets. Later they are called in a /code/models/train.py.
Also DVC is used to keep track of data versions.
<br>
<h2><i>#TODO wrap the opeartions into Airflow tasks</i></h2>

<h1>Stage 2. Model engineering</h1>
<h2>Input:</h2>
<ul>- Processed data</ul>
<h2>Output:</h2>
<ul>
- Trained model file<br>
- Model evaluation
</ul>

Model training, evaluation and packaging are performed by /code/models/train.py
Also MLflow is used for logging.
<br>
<h2><i>#TODO wrap the opeartions into Airflow tasks</i></h2>

<h1>Stage 3. Deployment</h1>
<h2>Input:</h2>
<ul>- Trained model</ul>
<h2>Output:</h2>
<ul>
- Running model API<br>
- Running app
</ul>

API calls the function from /code/models/predict.py when it recieves
a request GET/predict/?title={title}&text={text} from the Streamlit app.
FastAPI and Streamlit apps are placed in separate containers that communicate
according to the rules defined in /docker-compose.yaml
<br>
<h2><i>#TODO wrap the docker-compose into Airflow tasks</i></h2>
