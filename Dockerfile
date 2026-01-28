# syntax=docker/dockerfile:1

# Paperflow - Smart paper sorting and summarization for Zotero
# Multi-stage build using uv for fast dependency installation

FROM python:3.13-slim AS builder

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Set up working directory
WORKDIR /app

# Copy dependency files first for better caching
COPY pyproject.toml uv.lock README.md ./

# Install dependencies into a virtual environment
RUN uv sync --frozen --no-dev --no-install-project

# Copy the rest of the application
COPY src/ ./src/
COPY prompts/ ./prompts/

# Install the project itself
RUN uv sync --frozen --no-dev


# Runtime stage
FROM python:3.13-slim

WORKDIR /app

# Copy the virtual environment from builder
COPY --from=builder /app/.venv /app/.venv

# Copy application files
COPY --from=builder /app/src ./src
COPY --from=builder /app/prompts ./prompts

# Set PATH to use the virtual environment
ENV PATH="/app/.venv/bin:$PATH"

# Create directories for mounted volumes
RUN mkdir -p /app/.logs /app/.cache

# Default interval: 300 seconds (5 minutes)
ENV PAPERFLOW_INTERVAL=300

# Run the daemon
# Config and .env will be mounted at runtime
CMD ["sh", "-c", "paperflow start --interval ${PAPERFLOW_INTERVAL}"]
