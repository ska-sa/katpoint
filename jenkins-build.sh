#!/bin/bash
set -e -x
pip install -U pip setuptools wheel
pip install coverage
install-requirements.py -d ~/docker-base/base-requirements.txt -r requirements.txt
# install-requirements.py -d ~/docker-base/base-requirements.txt -r test-requirements.txt
nosetests --with-xunit --with-coverage --cover-erase --cover-xml --cover-inclusive --cover-package=katpoint
