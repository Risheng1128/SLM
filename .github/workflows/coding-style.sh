#!/usr/bin/env bash

# download pycodestyle
pip3 install pycodestyle

SOURCES=$(find $(git rev-parse --show-toplevel) | egrep "\.(py)\$")

set -x

for file in ${SOURCES};
do
    if [ ! -z "$(pycodestyle --config=.pycodestyle $file)" ]
    then
        exit "1"
    fi
done