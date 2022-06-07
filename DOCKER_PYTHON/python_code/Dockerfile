FROM python3.8

RUN mkdir /python_code

COPY requirements.txt /python_code

WORKDIR /python_code

RUN pip install -r requirements.txt 

COPY . /python_code

EXPOSE 8000

CMD ["uvicorn", "server:app", "--host", "127.0.0.1", "--port", "8000"]