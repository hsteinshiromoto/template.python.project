# ---
# Build arguments
# ---

ARG DOCKER_PARENT_IMAGE

FROM $DOCKER_PARENT_IMAGE

# NB: Arguments should come after FROM otherwise they're deleted
#ARG USER

# Silence debconf
ARG DEBIAN_FRONTEND=noninteractive

ENV TINI_VERSION v0.6.0

# ---
# Add Tini. Tini operates as a process subreaper for jupyter. This prevents
# kernel crashes.
# ---
USER root
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /usr/bin/tini
RUN chmod +x /usr/bin/tini
ENTRYPOINT ["/usr/bin/tini", "--"]


# Set container time zone

RUN pip install jupyter

LABEL maintainer="Dr Humberto STEIN SHIROMOTO <h.stein.shiromoto@gmail.com>"

#RUN useradd -ms /bin/bash $USER

# Create the "home" folder
RUN mkdir -p $PROJECT_DIR
WORKDIR $PROJECT_DIR

EXPOSE 8888

USER $USER
CMD su - user -c "jupyter notebook --no-browser --ip=0.0.0.0 --port=8888"
#CMD ["jupyter", "notebook", "--no-browser", "--ip=0.0.0.0", "--allow-root", "--port=8888"]