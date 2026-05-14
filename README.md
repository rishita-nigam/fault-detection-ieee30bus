# Fault Detection and Classification in IEEE 30-Bus Power System

## Overview
ML-based fault classification system for power systems.
Detects and classifies 5 fault types: Normal, LG, LL, LLG, and LLL
across the IEEE 30-bus test network.

## Results
- 87.5% test accuracy | 89% cross-validation accuracy
- 5 fault classes classified
- Permutation-based XAI used to validate fault indicators physically

## Tech Stack
Python | MATLAB Simulink | scikit-learn | pandas | NumPy | Matplotlib | Seaborn

## How It Works
1. 1000 fault scenarios simulated in MATLAB Simulink (Simscape Power Systems)
2. Voltage/current features extracted per fault event
3. Random Forest and SVM classifiers trained on extracted features
4. Permutation importance used to identify physically significant features
5. Predictions validated by re-running Simulink per sample

## Project Structure
- classifier/ — model training and evaluation scripts
- data/ — sample dataset (full dataset from Simulink simulation)
- results/ — confusion matrix, accuracy plots, feature importance

## How to Run
pip install -r requirements.txt
python classifier/train_model.py
python classifier/evaluate.py

## Background
The IEEE 30-bus system is a standard power systems test network.
Fault classification is critical for protective relay coordination
and grid stability. This project applies ML to automate what
traditionally requires expert manual analysis.

## Contact
Rishita Nigam | rishitanigam14@gmail.com
