#!/bin/bash

## Test python environment is setup correctly
if [[ $1 = "test_environment" ]]; then
	echo ">>> Testing Python Environment"
	python test_environment.py
fi

## Install Python Dependencies
if [[ $1 = "requirements" ]]; then
 	bash run_python.sh test_environment

 	echo ">>> Installing Required Modules .."
	python -m pip install -U pip setuptools wheel
	python -m pip install -r requirements.txt
	echo ">>> Done!"
fi

## Make Dataset
if [[ $1 == "data" ]]; then
	bash run_python.sh requirements
	python src/data/make_dataset.py data/raw data/processed
fi

## Delete all compiled Python files
if [[ $1 = "clean" ]]; then
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete
fi

# ## Lint using flake8
# lint:
# 	flake8 src

# ## Upload Data to S3
# sync_data_to_s3:
# ifeq (default,$(PROFILE))
# 	aws s3 sync data/ s3://$(BUCKET)/data/
# else
# 	aws s3 sync data/ s3://$(BUCKET)/data/ --profile $(PROFILE)
# endif

# ## Download Data from S3
# sync_data_from_s3:
# ifeq (default,$(PROFILE))
# 	aws s3 sync s3://$(BUCKET)/data/ data/
# else
# 	aws s3 sync s3://$(BUCKET)/data/ data/ --profile $(PROFILE)
# endif

## Set up python interpreter environment
# Todo: test this!
if [[ $1 = "create_environment" ]]; then

	if [[ $(shell which conda) = True ]]; then
		@echo ">>> Detected conda, creating conda environment."
		conda create --name ${PROJECT_NAME} python=3
		@echo ">>> New conda env created. Activate with:\nsource activate ${PROJECT_NAME}"

	else
		python3 -m pip install -q virtualenv virtualenvwrapper
		@echo ">>> Installing virtualenvwrapper if not already intalled.\nMake sure the following lines are in shell startup file\n\
		export WORKON_HOME=$$HOME/.virtualenvs\nexport PROJECT_HOME=$$HOME/Devel\nsource /usr/local/bin/virtualenvwrapper.sh\n"
		@bash -c "source `which virtualenvwrapper.sh`;mkvirtualenv $(PROJECT_NAME) --python=$(PYTHON_INTERPRETER)"
		@echo ">>> New virtualenv created. Activate with:\nworkon $(PROJECT_NAME)"
	fi

fi
