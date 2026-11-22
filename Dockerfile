FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 MPLBACKEND=Agg
WORKDIR /app
COPY pyproject.toml ./
COPY src ./src
COPY vendor ./vendor
RUN pip install -e . && pip uninstall -y opencv-python && pip install "opencv-python-headless>=4.9,<5"
ENV PYTHONPATH=/app/src:/app/vendor COMPOSITION_MODELS=/models
# Models are fetched once at build time and checksum-verified; analysis makes no network requests.
RUN python -m composition_reader.cli install-model
CMD ["python","-m","composition_reader.site"]
