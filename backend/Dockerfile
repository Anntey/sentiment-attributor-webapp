FROM python:3.8-slim

WORKDIR /app

COPY . .

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

RUN python3 -m spacy download en_core_web_sm
RUN python3 manage.py migrate

EXPOSE 8000

CMD ["gunicorn", "MainApp.wsgi"]