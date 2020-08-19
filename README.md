<h1 align="center">Sentiment Attributor Web App</h1>
<p align="center"><img src="app.png" alt="image" /></p>

## To-do
- [x] Docker Compose
- [x] CI/CD: GitHub Actions
- [ ] AWS: EC2

## Backend
RESTful API made with Django that serves a PyTorch model at http://localhost:8000/api/predict. Given an input sentence, it predicts the probability the sentiment is positive. In addition, it gives attributions for how much each token affects the sentiment. It uses [Captum's](https://captum.ai/docs/algorithms#integrated-gradients) implementation of integrated gradients[<sup>[1]</sup>](https://arxiv.org/abs/1703.01365).

Main functionality is in backend/PredictionApp/[views.py](https://github.com/Anntey/sentiment-attributor-webapp/blob/master/backend/PredictionApp/views.py) and backend/PredictionApp/[apps.py](https://github.com/Anntey/sentiment-attributor-webapp/blob/master/backend/PredictionApp/apps.py). A notebook related to prototyping is in backend/PredictionApp/training/[train.ipynb](https://github.com/Anntey/sentiment-attributor-webapp/blob/master/backend/PredictionApp/training/train.ipynb)

Uses Gunicorn as a WSGI and NginX as a web server and reverse proxy.


## Frontend
A minimal React frontend at http://localhost:3000 that makes POST requests to the backend when the text form is submitted. The poster of a random movie is given as inspiration. The predicted probability is rendered and words in the original sentence are colored red-green by their corresponding attributions.

Main functionality is in frontend/src/[app.js](https://github.com/Anntey/sentiment-attributor-webapp/blob/master/frontend/src/app.js).

Uses [Serve](https://www.npmjs.com/package/serve) to server the /frontend/build/ folder.

## Run
Images for the backend and frontend are pulled from Dockerhub so only the [/nginx/](https://github.com/Anntey/sentiment-attributor-webapp/blob/master/nginx/) folder is needed.

```zsh
$ docker-compose up
```

Access and error logs from Gunicorn are saved to /logs/gunicorn and from NginX to /logs/nginx/.