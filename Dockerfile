# ---
# Build arguments
# ---
ARG DOCKER_PARENT_IMAGE="python:3.9-slim"
FROM $DOCKER_PARENT_IMAGE

# NB: Arguments should come after FROM otherwise they're deleted
ARG BUILD_DATE
ARG PROJECT_NAME

# Silence debconf
ARG DEBIAN_FRONTEND=noninteractive

# ---
# Enviroment variables
# ---
ENV LANG=C.UTF-8 \
    LC_ALL=C.UTF-8
ENV PROJECT_ROOT /home/$PROJECT_NAME
ENV PYTHONPATH $PROJECT_ROOT
ENV TZ Australia/Sydney
ENV JUPYTER_ENABLE_LAB=yes


# Set container time zone
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

LABEL org.label-schema.build-date=$BUILD_DATE \
      maintainer="Dr Humberto STEIN SHIROMOTO <h.stein.shiromoto@gmail.com>"

# ---
# Set up the necessary Debian packages
# ---
COPY debian-requirements.txt /usr/local/debian-requirements.txt

RUN apt-get update && \
	DEBIAN_PACKAGES=$(egrep -v "^\s*(#|$)" /usr/local/debian-requirements.txt) && \
    apt-get install -y $DEBIAN_PACKAGES && \
    apt-get clean

# ---
# Copy Container Setup Scripts
# ---

COPY bin/entrypoint.sh /usr/local/bin/entrypoint.sh
COPY bin/run_python.sh /usr/local/bin/run_python.sh
COPY bin/test_environment.py /usr/local/bin/test_environment.py
COPY bin/setup.py /usr/local/bin/setup.py
COPY requirements.txt /usr/local/requirements.txt

RUN chmod +x /usr/local/bin/entrypoint.sh && \
    chmod +x /usr/local/bin/run_python.sh && \
	chmod +x /usr/local/bin/test_environment.py && \
	chmod +x /usr/local/bin/setup.py

RUN bash /usr/local/bin/run_python.sh test_environment && \
	bash /usr/local/bin/run_python.sh requirements

# Create the "home" folder
RUN mkdir -p $PROJECT_ROOT
WORKDIR $PROJECT_ROOT

RUN useradd -ms /bin/bash  vscode
USER vscode
WORKDIR $PROJECT_ROOT

# ---
# Setup running and entrypoint
# ---
#Expose Jupyter port
EXPOSE 8888 
# CMD ["jupyter", "lab", "--port=8888", "--no-browser", "--ip=0.0.0.0", "--allow-root"]

EXPOSE 22
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
