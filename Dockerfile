FROM nvcr.io/nvidia/pytorch:25.08-py3

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Add /app as search path for py modules
ENV PYTHONPATH=/app
# No buffering when logging to stdout
ENV PYTHONUNBUFFERED=1

# App code
COPY codebase_indexer.py .
COPY mcp_server.py .

# Default command to start the MCP server
CMD ["python", "mcp_server.py"]
