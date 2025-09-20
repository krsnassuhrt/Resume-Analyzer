#!/usr/bin/env bash
set -e
pip install -r backend/requirements.txt
python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('averaged_perceptron_tagger', quiet=True)"