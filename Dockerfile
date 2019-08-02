# ---
# Build arguments
# ---

ARG DOCKER_PARENT_IMAGE

FROM $DOCKER_PARENT_IMAGE

# NB: Arguments should come after FROM otherwise they're deleted
ARG BUILD_DATE
ARG PROJECT_NAME
ARG DOCKER_IMAGE
ARG REGISTRY
ARG FILES
ARG USER

# Setup AWS S3 access
ARG AWS_ACCESS_KEY_ID
ARG AWS_SECRET_ACCESS_KEY

# Silence debconf
ARG DEBIAN_FRONTEND=noninteractive

# ---
# Enviroment variables
# ---
ENV LANG=C.UTF-8 \
    LC_ALL=C.UTF-8
ENV PROJECT_DIR /home/$PROJECT_NAME
ENV PYTHONPATH $PROJECT_DIR
ENV DOCKER_IMAGE $DOCKER_IMAGE
ENV REGISTRY $REGISTRY
ENV TZ Australia/Sydney

# Setup AWS S3 access
ENV AWS_ACCESS_KEY_ID $AWS_ACCESS_KEY_ID
ENV AWS_SECRET_ACCESS_KEY $AWS_SECRET_ACCESS_KEY

# Set container time zone
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

LABEL org.label-schema.build-date=$BUILD_DATE \
      maintainer="Dr Humberto STEIN SHIROMOTO <h.stein.shiromoto@gmail.com>"

# ---
# Set up the necessary Debian packages
# ---
#RUN apt update
#RUN apt install -y git procps cron sudo groff

RUN useradd -ms /bin/bash $USER

# Create the "home" folder
RUN mkdir -p $PROJECT_DIR
WORKDIR $PROJECT_DIR

# ---
# Set up the necessary Python environment and packages
# ---
COPY "run_python.sh" "test_environment.py" "setup.py" $FILES $PROJECT_DIR/
RUN bash run_python.sh test_environment
RUN bash run_python.sh requirements

USER $USER

