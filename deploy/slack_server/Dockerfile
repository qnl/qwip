FROM python:3.10-bullseye as base

FROM base as builder

RUN mkdir /install
WORKDIR /install

COPY requirements.txt /requirements.txt

RUN pip install --prefix=/install -r /requirements.txt

FROM base

COPY --from=builder /install /usr/local
COPY src/qwip_slack /qwip_slack

ENTRYPOINT ["gunicorn", "-w 4", "--worker-class", "uvicorn.workers.UvicornWorker", "qwip_slack.server:app", "--bind", "0.0.0.0:4001"]
