# ===== Base: CUDA 12.8 + Ubuntu 24.04 =====
FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    LANG=C.UTF-8 LC_ALL=C.UTF-8

# Core OS deps + headless GL (Ubuntu 24.04: use libgl1, NOT libgl1-mesa-glx)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git git-lfs wget curl vim ca-certificates bzip2 build-essential cmake \
    libglib2.0-0 libsm6 libxext6 libxrender1 \
    libosmesa6-dev libgl1 libglx-mesa0 libglfw3 patchelf \
    ffmpeg tini \
 && rm -rf /var/lib/apt/lists/* && git lfs install

# ===== Install Miniforge (conda-forge) =====
RUN wget -q https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O /tmp/miniforge.sh \
 && bash /tmp/miniforge.sh -b -p /opt/conda \
 && rm /tmp/miniforge.sh
ENV PATH=/opt/conda/bin:$PATH
RUN conda config --set channel_priority flexible && conda config --add channels conda-forge

# ===== Create env =====
RUN conda create -y -n openx python=3.11 pip && conda clean -afy

# ===== Clone repo into /workspace/demonstration-information =====
WORKDIR /workspace
RUN git clone https://github.com/jhejna/demonstration-information.git /workspace/demonstration-information
WORKDIR /workspace/demonstration-information

# ===== Install Python deps & package =====
RUN conda run -n openx pip install --upgrade pip && \
    conda run -n openx pip install -r requirements.txt && \
    conda run -n openx pip install -e .

# ===== JAX (CUDA 12 by default, switchable to CPU) =====
ARG JAX_MODE=cuda12
RUN if [ "$JAX_MODE" = "cuda12" ]; then \
      conda run -n openx pip install --upgrade "jax[cuda12_pip]==0.4.37" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html ; \
    else \
      conda run -n openx pip install --upgrade "jax[cpu]==0.4.37" ; \
    fi

# ===== Optional: Robomimic toolchain (MuJoCo-era) =====
# Enable with: --build-arg WITH_ROBOMIMIC=1
# NOTE: Ubuntu 24.04 keeps libglx-mesa0 (GLX vendor lib) and libgl1 (GLVND). :contentReference[oaicite:2]{index=2}
# For mujoco-py you still need the old mujoco210 binaries at runtime — we avoid building mujoco-py here.
ARG WITH_ROBOMIMIC=0
RUN if [ "$WITH_ROBOMIMIC" = "1" ]; then \
      echo "[Robomimic stack] installing robosuite & robomimic (skip mujoco-py build)"; \
      git clone https://github.com/ARISE-Initiative/robosuite.git /opt/robosuite && \
      cd /opt/robosuite && git checkout offline_study && conda run -n openx pip install -e . --no-deps && \
      git clone https://github.com/ARISE-Initiative/robomimic.git /opt/robomimic && \
      cd /opt/robomimic && git checkout v0.2.0 && conda run -n openx pip install -e . --no-deps && \
      printf '%s\n' '#!/usr/bin/env bash' \
        'set -euo pipefail' \
        'source /opt/conda/etc/profile.d/conda.sh' \
        'conda activate openx' \
        'echo "To build mujoco-py you must provide mujoco210 (old MuJoCo) at ~/.mujoco/mujoco210 and then run:"' \
        'echo "  MUJOCO_PY_MUJOCO_PATH=$HOME/.mujoco/mujoco210 pip install '"'"'mujoco-py<2.2,>=2.0'"'"' cython==0.29.37 numba"' \
        > /usr/local/bin/install_mujoco_py.sh && chmod +x /usr/local/bin/install_mujoco_py.sh ; \
    fi

# ===== Make setup_shell.sh run on every interactive shell =====
ENV PYTHONPATH=/workspace/demonstration-information
SHELL ["/bin/bash", "-lc"]

# Ensure the script is executable
RUN chmod +x /workspace/demonstration-information/setup_shell.sh || true

# System-wide login shell hooks:
# 1) initialize conda
# 2) activate env
# 3) cd to /workspace
# 4) source the project's setup_shell.sh (non-fatal if user later edits it)
RUN echo 'source /opt/conda/etc/profile.d/conda.sh' >> /etc/bash.bashrc && \
    echo 'conda activate openx' >> /etc/bash.bashrc && \
    echo 'cd /workspace' >> /etc/bash.bashrc && \
    echo '[ -f /workspace/demonstration-information/setup_shell.sh ] && . /workspace/demonstration-information/setup_shell.sh || true' >> /etc/bash.bashrc

# ===== Startup wrapper (login shell, repo present, setup sourced) =====
RUN printf '%s\n' '#!/usr/bin/env bash' \
    'set -e' \
    'source /opt/conda/etc/profile.d/conda.sh' \
    'conda activate openx' \
    'if [ ! -d "/workspace/demonstration-information/.git" ]; then' \
    '  git clone https://github.com/jhejna/demonstration-information.git /workspace/demonstration-information || true' \
    'fi' \
    'cd /workspace' \
    '# Run the project shell setup once (ignore noncritical failures)' \
    'if [ -f /workspace/demonstration-information/setup_shell.sh ]; then' \
    '  set +e; . /workspace/demonstration-information/setup_shell.sh; set -e' \
    'fi' \
    'exec bash -l' > /usr/local/bin/start.sh && chmod +x /usr/local/bin/start.sh

ENTRYPOINT ["/usr/local/bin/start.sh"]

