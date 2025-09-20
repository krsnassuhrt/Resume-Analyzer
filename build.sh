#!/usr/bin/env bash

exit on error
set -o errexit

pip install -r backend/requirements.txt

python -c "import nltk; nltk.download('punkt'); nltk.download('averaged_perceptron_tagger')"