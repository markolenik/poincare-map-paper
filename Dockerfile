FROM python:3.8.6

# Add 32 bit support to run xppaut
RUN dpkg --add-architecture i386

# The texlive stuff is necessary for using the pgf matplotlib backend.
RUN apt-get update && \
    apt-get install -y libx11-6:i386 vim less \
    texlive texlive-luatex texlive-science texlive-latex-extra \
    texlive-fonts-extra

# Install xpp
RUN curl -O http://www.math.pitt.edu/~bard/bardware/binary/latest/xpplinux.tgz \
    && tar -xzf xpplinux.tgz

RUN cp /xppaut8.0ubuntu/xppaut /bin/.

# Create user
ARG USER=root
ARG UID=1000
ARG GID=1000

RUN groupadd -g ${GID} -o ${USER}
RUN useradd -m -s /bin/bash -u ${UID} -g ${GID} -o ${USER}
USER ${USER}
ENV HOME=/home/${USER}
WORKDIR ${HOME}

# Enable bash syntax colouring.
ENV TERM xterm-256color

# Install poetry
RUN curl -sSL \
    https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py \
    | python
ENV PATH = "${PATH}:${HOME}/.poetry/bin"

ENV APP_DIR=${HOME}/app
WORKDIR ${APP_DIR}
# Add source folder to pythonpath for direct python shell execution. Have to
# do this since we will be mounting source folder only after 'poetry install',
# wich won't add that package to the path otherwise.
ENV PYTHONPATH=${APP_DIR}
COPY pyproject.toml poetry.lock ./
RUN poetry install

# Add autoreload jupyter extension and switch off interactive plotting.
RUN echo "c.InteractiveShellApp.exec_lines = ['%load_ext autoreload', \
    '%autoreload 2', \
    'import matplotlib.pyplot as plt', \
    'plt.ioff()']" \
    > "$(poetry run ipython locate)/profile_default/ipython_config.py"

# RUN echo "c.InteractiveShellApp.exec_lines = ['%load_ext autoreload', '%autoreload 2']" \
#     > "$(poetry run ipython locate)/profile_default/ipython_config.py"
