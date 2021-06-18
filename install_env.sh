#!/usr/bin/env bash
conda create -n myenv python=3.8
conda activate myenv
pip install -r requirements.txt
python -m spacy download en_core_web_sm
