#!/bin/bash
set -e
echo "=== Starting entrypoint script ==="

# Install Unsloth
echo "Installing Unsloth..."
INSTALL_CMD=$(wget -qO- https://raw.githubusercontent.com/unslothai/unsloth/main/unsloth/_auto_install.py | python -)
echo "Running installation command: $INSTALL_CMD"
eval "$INSTALL_CMD"

# Inverted logic: default is bash, and only if ENABLE_JUPYTER=true will Jupyter run
if [ "${ENABLE_JUPYTER}" = "true" ]; then
    # Check if we should use JupyterLab or Notebook
    if [ "${USE_JUPYTERLAB}" = "true" ]; then
        echo "Starting JupyterLab server on port 8888"
        pip install jupyterlab -q
        echo "JupyterLab installation complete"
        echo "Access JupyterLab by connecting to http://<host-ip>:8888"
        jupyter lab --ip=0.0.0.0 --port=8888 --allow-root --no-browser
    else
        echo "Starting Jupyter notebook server on port 8888"
        pip install jupyter -q
        echo "Jupyter installation complete"
        echo "Access the notebook by connecting to http://<host-ip>:8888"
        jupyter notebook --ip=0.0.0.0 --port=8888 --allow-root --no-browser
    fi
else
    echo "ENABLE_JUPYTER is not set to true, returning to bash shell"
    exec "$@"  # Execute the CMD (which is /bin/bash as defined in Dockerfile)
fi