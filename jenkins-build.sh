#!/bin/bash
set -e -x
pip install -r ~/docker-base/pre-requirements.txt
install-requirements.py -d ~/docker-base/base-requirements.txt -r requirements.txt -r test-requirements.txt
nosetests --with-xunit --with-coverage --cover-erase --cover-xml --cover-inclusive --cover-package=katpoint
