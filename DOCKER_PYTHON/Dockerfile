

FROM ubuntu:latest

RUN apt-get update

RUN apt-get install -y python3 python3-pip

RUN python3 --version

RUN mkdir /predict_churn

COPY ./python_code/requirements.txt /predict_churn

VOLUME /predict_churn

RUN pip3 install -r /predict_churn/requirements.txt



ENTRYPOINT [ "python3" ]

CMD [ "/predict_churn/my_first_api.py" ]