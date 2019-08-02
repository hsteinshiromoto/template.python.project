# ---
# Build arguments
# ---

ARG DOCKER_PARENT_IMAGE

FROM $DOCKER_PARENT_IMAGE

# NB: Arguments should come after FROM otherwise they're deleted
ARG BUILD_DATE
ARG PROJECT_NAME
ARG REGISTRY
ARG FILES
ARG USER

# Silence debconf
ARG DEBIAN_FRONTEND=noninteractive

# ---
# Enviroment variables
# ---
ENV LANG=C.UTF-8 \
    LC_ALL=C.UTF-8
ENV PROJECT_DIR /home/$PROJECT_NAME
ENV PYTHONPATH $PROJECT_DIR
ENV REGISTRY $REGISTRY
ENV TZ Australia/Sydney

# Set container time zone
USER root
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

LABEL org.label-schema.build-date=$BUILD_DATE \
      maintainer="Dr Humberto STEIN SHIROMOTO <h.stein.shiromoto@gmail.com>"

RUN useradd -ms /bin/bash $USER

# Create the "home" folder
RUN mkdir -p $PROJECT_DIR
WORKDIR $PROJECT_DIR

EXPOSE 8888