#!/bin/bash
TMPDIR=$(mktemp -d) && \
echo
echo "> Creating mount directory at ${TMPDIR}"
rsync -avc --exclude-from=./rsyncignore "${PWD}/" "${TMPDIR}" && \

echo
echo "> Start Jupyter container"
sudo docker run --rm \
	--cap-drop ALL --security-opt no-new-privileges \
	--user jupyteruser --cpus=12 \
	-v "$TMPDIR":/home/jupyteruser/work \
	-p 127.0.0.1:8888:8888 jupyter-qtime && \

echo
echo "> Sync changes"
rsync -avc --exclude-from=./rsyncignore "${TMPDIR}/" "${PWD}" && \
rm -rf "${TMPDIR}"

echo
echo "> Completed"
