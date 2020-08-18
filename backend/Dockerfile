FROM python:3.7

WORKDIR /src

COPY . .

#RUN pip install torch torchvision torchtext spacy Django djangorestframework django-cors-headers python-dotenv captum gunicorn
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

RUN python -m spacy download en_core_web_sm
RUN python manage.py migrate
