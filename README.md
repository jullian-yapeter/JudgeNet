# JudgeNet
Neural Network to rate a speaker based on what was said and how it was said

# Quick Start
git clone this repo and `cd` into the `judgenet` root directory
```
git clone git@github.com:jullian-yapeter/JudgeNet.git && cd JudgeNet
python3 -m venv .env
source .env/bin/activate
python3 -m pip install -e .
cd .. && git clone git@github.com:pytorch/rl.git && cd rl
python3 -m pip install -e .
cd ../JudgeNet
python3 judgenet/main.py
```
