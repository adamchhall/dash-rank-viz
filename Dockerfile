FROM python:3.8

WORKDIR /

COPY . /

RUN pip install --trusted-host pypi.python.org -r requirements.txt

EXPOSE 8080

CMD ["python", "main.py"]