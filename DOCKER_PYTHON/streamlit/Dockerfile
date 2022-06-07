

FROM python:3.8

RUN mkdir /streamlit

COPY requirements.txt /streamlit

WORKDIR /streamlit

RUN pip3 install -r requirements.txt

COPY . /streamlit

EXPOSE 8501

CMD ["streamilt", "run", "ui.py"]