FROM python:3.13-slim

# Create user and directory
RUN useradd -ms /bin/bash jupyteruser
USER jupyteruser
WORKDIR /home/jupyteruser/work

# Install
COPY requirements.txt .

USER root
RUN apt update && apt install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && pip install --no-cache-dir -r requirements.txt \
    && apt autoremove --purge \
    && apt clean \
    && rm -rf /var/lib/apt/lists/*

# Expose Jupyter
USER jupyteruser
EXPOSE 8888

# Start
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--no-browser"]
