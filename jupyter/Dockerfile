# ---
# Build arguments
# ---

ARG DOCKER_PARENT_IMAGE

FROM $DOCKER_PARENT_IMAGE

# NB: Arguments should come after FROM otherwise they're deleted
ARG USER

# Silence debconf
ARG DEBIAN_FRONTEND=noninteractive

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