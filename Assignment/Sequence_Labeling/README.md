In this assignment, you will implement the fundamental deep learning models for the task of sequence labeling. Sequence labeling has many applications in NLP, including named entity recognition (NER), part of speech tagging, and slot filling. Essentially, given a sentence, this task aims to provide a label for each word in the sentence to capture some specific information (i.e., names, verbs, nouns). The usual approach to encode the labels is based on the BIO tagging schema (Begin, Inside, Other).

To learn more about sequence labeling and NER, you can read this paper and its related work: https://arxiv.org/pdf/1603.01354.pdf 

Training, test, and development data for this assignment can be found in this folder, where x.text and x.label (x = train, dev or test) are the raw text file and its corresponding label file.

Each line in x.text is a sentence that should be separated by spaces to obtain the list of words. Each line d in x.text will have one corresponding line l (in the same order) in x.label that involves a sequence of labels for the words d.

For instnace, the first sentence in the file train.text is: 

```listen to westbam alumb allergic on google music```

The corresponding label sequence for this sentence in the first line of train.label is:

```O     O B-artist   O   B-album  O B-service I-service```

This label sequence indicates that "westbam" is an artist, "allergic" is an album, and "google music" is a service.

At the mimimum, you will need to implement the following models for this dataset to complete the assignment:

1. A bidirectional LSTM model using either word2vec or Glove as the input word embeddings;
2. A BERT-based model;
3. A conditional random field (CRF) layer that is put on top of the above models (see the paper above);

You are encouraged to try other models for this problem, including convolutional neural networks, self-attention, ELMo embeddings, and character-based embeddings.

For each model, tune it as much as you want with the development files (dev.text and dev.label). Once you are satistified, evaluate the final model on the test data. Never tune your models on test data. Report the performance of each implemented model on the development and test data.

You can use the provided scorer file "seq_scorer.py" to score your model. To use this scorer, your program should produce output files with the same format as the *.label files (one sequence label in each line for each sentence in the *.text files). Afterward, you can run the following command to obtain the scores:

```
python seq_scorer.py path_to_the_golden_label_file path_to_the_predicted_label_file
```

If things run correctly, you will see the precision, recall, and F1 scores for your model. use these scores as the performance for reporting.

You will need to install scikit-learn for this using:

```pip install -U scikit-learn```
