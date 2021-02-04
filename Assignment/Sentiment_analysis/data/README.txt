-- Data format: 

+ each line has the format: #id + SPACE + sentence + SPACE + #label

+ two labels: pos and neg.

-- Evaluation script: 

+ syntax: python Evaluation_script.py --pred-file path-to-predict-file --gt-file path-to-ground-truth-file

+ each line in the prediction file follows the format: #id + SPACE + #label

+ output: triple (accuracy, macro_f1, micro_f1)