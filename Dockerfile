FROM python:3.8

WORKDIR /app

RUN pip install -r requirements.txt

COPY model.py .
COPY prepare_data.py .
COPY train_model.py .
COPY anime.csv .
COPY anime_test.csv .

CMD [ "python", "train_model.py" ]