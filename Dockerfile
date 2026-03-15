FROM public.ecr.aws/lambda/python:3.11

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=${LAMBDA_TASK_ROOT}

WORKDIR ${LAMBDA_TASK_ROOT}

COPY requirements.txt .
RUN pip install --no-cache-dir --prefer-binary -r requirements.txt

COPY app.py .
COPY notify.py .
COPY models ./models
COPY agents ./agents
COPY data ./data

CMD ["app.handler"]