#!/bin/bash
set -e -x
pip install -r ~/docker-base/pre-requirements.txt
pip install -r requirements.txt -r test-requirements.txt
nosetests --with-xunit --with-coverage --cover-erase --cover-xml --cover-inclusive --cover-package=katpoint
