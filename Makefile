################################################
#				ASSUMPTIONS
################################################
## If you have `pyenv` and `pyenv-virtualenv`
## installed beforehand, run the `darwin`
## or `linux` commands.

SHELL := /bin/bash


darwin: ## >> pre-requirements for darwin
	brew update
	brew install pyenv pyenv-virtualenv

linux: ## >> pre-requirements for linux
	pip install pyenv

################################################
#				MAIN COMMANDS
################################################
PROJECT=transformer-experiments
VERSION=3.8.2
VENV=${PROJECT}-${VERSION}
VENV_DIR=$(shell pyenv root)/versions/${VENV}
PYTHON=${VENV_DIR}/bin/python


# Colors for echos
ccend=$(shell tput sgr0)
ccbold=$(shell tput bold)
ccgreen=$(shell tput setaf 2)
ccso=$(shell tput smso)

clean: ## >> remove all environment and build files.
	@echo ""
	@echo "$(ccso)--> Removing virtual environment $(ccend)"
	pyenv virtualenv-delete --force ${VENV}
	rm .python-version

build: ##@main >> build a new virtual environment.
	@echo ""
	@echo "$(ccso)--> Build $(ccend)"
	$(MAKE) install

venv: $(VENV_DIR)

$(VENV_DIR):
	@echo "$(ccso)--> Install and setup pyenv and virtualenv $(ccend)"
	pyenv install --skip-existing ${VERSION}
	pyenv virtualenv ${VERSION} ${VENV}
	echo ${VENV} > .python-version

install: venv requirements.txt ##@main >> update requirements.txt inside the venv
	@echo "$(ccso)--> Updating packages $(ccend)"
	$(PYTHON) -m pip install -r requirements.txt

activate: venv
	pyenv activate ${VENV}

deactivate: ##@main >> deactivate current virtualenv and use local system python
	pyenv deactivate ${VENV}

################################################
#				HELPERS
################################################

# And add help text after each target name starting with '\#\#'
# A category can be added with @category
HELP_FUNC = \
	%help; \
	while(<>) { push @{$$help{$$2 // 'options'}}, [$$1, $$3] if /^([a-zA-Z\-\$\(]+)\s*:.*\#\#(?:@([a-zA-Z\-\)]+))?\s(.*)$$/ }; \
	print "usage: make [target]\n\n"; \
	for (sort keys %help) { \
	print "${WHITE}$$_:${RESET}\n"; \
	for (@{$$help{$$_}}) { \
	$$sep = " " x (32 - length $$_->[0]); \
	print "  ${YELLOW}$$_->[0]${RESET}$$sep${GREEN}$$_->[1]${RESET}\n"; \
	}; \
	print "\n"; }

help: ##@other >> Show this help.
	@perl -e '$(HELP_FUNC)' $(MAKEFILE_LIST)
	@echo ""
	@echo "Note: to activate the environment in your local shell type:"
	@echo "   $$ pyenv activate $(VENV)"