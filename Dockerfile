FROM ubuntu:latest

RUN mkdir -p /workspace
WORKDIR /workspace
COPY ./pyproject.toml /workspace
COPY ./README.md /workspace
COPY ./src /workspace/src

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    make \
&& apt-get clean

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
ENV UV_LINK_MODE=copy
RUN uv venv
RUN uv sync --all-extras

CMD [ "bash" ]