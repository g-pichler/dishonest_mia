# syntax=docker/dockerfile:1.3

FROM python:3.9-slim-bullseye

RUN useradd -u 1000 -ms /bin/bash defaultuser

USER defaultuser
WORKDIR /home/defaultuser
ENV VIRTUAL_ENV=./.venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# default port
ENV port=8081

# Copy everything:
COPY --chown=1000 ./ ./
# Create cache
RUN mkdir ./.cache

# Install dependencies:
RUN --mount=type=cache,target=/home/defaultuser/.cache/pip,uid=1000 pip install -r requirements.txt

CMD jupyter notebook --port=${port} --no-browser --ip=0.0.0.0
