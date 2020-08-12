<h1 align="center">Sentiment Attributor Web App</h1>
<p align="center"><img src="app.png" alt="image" /></p>

## Backend
RESTful API made with Django that serves a PyTorch model at http://localhost:8000/api/predict. Given an input sentence, it predicts the probability the sentiment is positive. In addition, it gives attributions for how much each token affects the sentiment. It uses [Captum's Integrated Gradients](https://captum.ai/docs/algorithms#integrated-gradients).

```zsh
$ python manage.py runserver
```

## Frontend
A minimal React frontend at http://localhost:3000 that makes POST requests to the backend when the text form is submitted. The predicted probability is rendered and words in the original sentence are colored red-green by their corresponding attributions.

```zsh
$ npm start
```

# To-do
- [ ] Gunicorn
- [ ] NGINX
- [ ] Docker
- [ ] AWS ECS
