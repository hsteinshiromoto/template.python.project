#!/bin/bash


## Test python environment is setup correctly
if [[ $1 = "test_environment" ]]; then
	${PYTHON_INTERPRETER} test_environment.py
fi

# Install Python Dependencies
if [[ $1 = "requirements" ]]; then
 	bash run_python.sh test_environment
	${PYTHON_INTERPRETER} -m pip install -U pip setuptools wheel
	${PYTHON_INTERPRETER} -m pip install -r requirements.txt
fi

# ## Make Dataset
# data: requirements
# 	$(PYTHON_INTERPRETER) src/data/make_dataset.py data/raw data/processed

# ## Delete all compiled Python files
# clean:
# 	find . -type f -name "*.py[co]" -delete
# 	find . -type d -name "__pycache__" -delete

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

# ## Set up python interpreter environment
# create_environment:
# ifeq (True,$(HAS_CONDA))
# 		@echo ">>> Detected conda, creating conda environment."
# ifeq (3,$(findstring 3,$(PYTHON_INTERPRETER)))
# 	conda create --name $(PROJECT_NAME) python=3
# else
# 	conda create --name $(PROJECT_NAME) python=2.7
# endif
# 		@echo ">>> New conda env created. Activate with:\nsource activate $(PROJECT_NAME)"
# else
# 	$(PYTHON_INTERPRETER) -m pip install -q virtualenv virtualenvwrapper
# 	@echo ">>> Installing virtualenvwrapper if not already intalled.\nMake sure the following lines are in shell startup file\n\
# 	export WORKON_HOME=$$HOME/.virtualenvs\nexport PROJECT_HOME=$$HOME/Devel\nsource /usr/local/bin/virtualenvwrapper.sh\n"
# 	@bash -c "source `which virtualenvwrapper.sh`;mkvirtualenv $(PROJECT_NAME) --python=$(PYTHON_INTERPRETER)"
# 	@echo ">>> New virtualenv created. Activate with:\nworkon $(PROJECT_NAME)"
# endif
