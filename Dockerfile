FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000

WORKDIR /srv/stripe-reconstructor

COPY pyproject.toml README.md ./
COPY app ./app
COPY tests ./tests
COPY scripts ./scripts
COPY verify.py ./

# The service itself runs on the standard library alone. The build
# toolchain below is only used by the verify stage's package build and
# is best-effort: scripts/package_build.py falls back to a stdlib
# builder when the network is unavailable.
RUN pip install --no-cache-dir "build>=1" "setuptools>=61" wheel || true

EXPOSE 8000

CMD ["python", "-m", "app"]
