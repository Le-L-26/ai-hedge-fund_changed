# Dockerfile for the my-ai-agent DATA BRIDGE (scripts/agent_bridge.py).
#
# This is the service that fetches real stock fundamentals (financialdatasets.ai
# for US/foreign tickers, akshare for Chinese A-shares) and serves them to the
# Nuxt agent over HTTP. Render (or any Docker host) builds this image and runs it
# as a web service.
#
# It reuses the existing src/tools/api.py data layer, so it installs the project's
# Python dependencies via Poetry, exactly like docker/Dockerfile does.

FROM python:3.11-slim

WORKDIR /app

# src.* imports resolve against /app
ENV PYTHONPATH=/app

# Poetry, pinned for reproducible builds
RUN pip install --no-cache-dir poetry==1.7.1

# Dependency files first for layer caching
COPY pyproject.toml poetry.lock* /app/

# Install into the system interpreter (no nested venv) so `python` sees the deps
RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi --no-root

# Application code
COPY . /app/

# The bridge reads $PORT (Render injects it) and binds 0.0.0.0; see the
# __main__ block in scripts/agent_bridge.py. EXPOSE is documentation only.
EXPOSE 8077
CMD ["python", "scripts/agent_bridge.py"]
