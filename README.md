# Natural Language Processing with Deep Learning

Materials extracted from the CS224n course "[Natural Language Processing with Deep Learning](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1194/)"

### Office hours 
- **Dat Quoc Nguyen**: 15:30-17:00 Fridays 25/12/2020-19/03/2021, Building K 

## Lectures

- **25/12/2020: Introduction and Word Vectors** [[slides](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1194/slides/cs224n-2019-lecture01-wordvecs1.pdf)] [[video](https://youtu.be/8rXD5-xhemo)] [[notes](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1194/readings/cs224n-2019-notes01-wordvecs1.pdf)]

	Suggested Readings:
	-  [Word2Vec Tutorial - The Skip-Gram Model](http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/)
	-  [Efficient Estimation of Word Representations in Vector Space](http://arxiv.org/pdf/1301.3781.pdf) (original word2vec paper)
	-  [Distributed Representations of Words and Phrases and their Compositionality](http://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf)  (negative sampling paper)
- **08/01/2021: Word Vectors 2 and Word Senses**  [[slides](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1194/slides/cs224n-2019-lecture02-wordvecs2.pdf)] [[video](https://youtu.be/kEMJRjEdNzM)] [[notes](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1194/readings/cs224n-2019-notes02-wordvecs2.pdf)]
	
	Suggested Readings:
	-  [GloVe: Global Vectors for Word Representation](http://nlp.stanford.edu/pubs/glove.pdf)  (original GloVe paper)
	-  [Improving Distributional Similarity with Lessons Learned from Word Embeddings](http://www.aclweb.org/anthology/Q15-1016)
	-  [Evaluation methods for unsupervised word embeddings](http://www.aclweb.org/anthology/D15-1036)
	
	Additional Readings:
	-  [A Latent Variable Model Approach to PMI-based Word Embeddings](http://aclweb.org/anthology/Q16-1028)
	-  [Linear Algebraic Structure of Word Senses, with Applications to Polysemy](https://transacl.org/ojs/index.php/tacl/article/viewFile/1346/320)
	-  [On the Dimensionality of Word Embedding](https://papers.nips.cc/paper/7368-on-the-dimensionality-of-word-embedding.pdf)

- **15/01/2021: Word Window Classification, Neural Networks, and Calculus** [[slides](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1194/slides/cs224n-2019-lecture03-neuralnets.pdf)] [[video](https://youtu.be/8CWyBNX6eDo)]  [[notes](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1194/readings/cs224n-2019-notes03-neuralnets.pdf)] [[matrix calculus notes](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1194/readings/gradient-notes.pdf)]  

	Suggested Readings:

	-  [CS231n notes on backprop](http://cs231n.github.io/optimization-2/)
	-  [Review of differential calculus](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1194/readings/review-differential-calculus.pdf)

	Additional Reading: [Natural Language Processing (Almost) from Scratch](http://www.jmlr.org/papers/volume12/collobert11a/collobert11a.pdf)

- **22/01/2021: The probability of a sentence? Recurrent Neural Networks and Language Models** [[slides](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1194/slides/cs224n-2019-lecture06-rnnlm.pdf)] [[video](https://youtu.be/iWea12EAu6U)]  [[notes](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1194/readings/cs224n-2019-notes05-LM_RNN.pdf)] [[assignment](https://github.com/VinAIResearch/DL4NLP/blob/master/Assignment/Sentiment%20analysis/Assignment%201%20-%20RNN.txt)]

	Suggested Readings:

	-  [N-gram Language Models](https://web.stanford.edu/~jurafsky/slp3/3.pdf)  (textbook chapter)
	-  [The Unreasonable Effectiveness of Recurrent Neural Networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness/) (blog post overview)
	-  [Sequence Modeling: Recurrent and Recursive Neural Nets](http://www.deeplearningbook.org/contents/rnn.html) (Sections 10.1 and 10.2)
	-  [On Chomsky and the Two Cultures of Statistical Learning](http://norvig.com/chomsky.html)

- **29/01/2021: Vanishing Gradients and Fancy RNNs**  [[slides](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1194/slides/cs224n-2019-lecture07-fancy-rnn.pdf)] [[video](https://youtu.be/QEw0qEa0E50)]  [[notes](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1194/readings/cs224n-2019-notes05-LM_RNN.pdf)]

	Suggested Readings:

	-  [Sequence Modeling: Recurrent and Recursive Neural Nets](http://www.deeplearningbook.org/contents/rnn.html) (Sections 10.3, 10.5, 10.7-10.12)
	-  [Learning long-term dependencies with gradient descent is difficult](http://ai.dinfo.unifi.it/paolo//ps/tnn-94-gradient.pdf)  (one of the original vanishing gradient papers)
	-  [On the difficulty of training Recurrent Neural Networks](https://arxiv.org/pdf/1211.5063.pdf)  (proof of vanishing gradient problem)
	-  [Vanishing Gradients Jupyter Notebook](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1174/lectures/vanishing_grad_example.html)  (demo for feedforward networks)
	-  [Understanding LSTM Networks](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)  (blog post overview)

- **05/02/2021: Machine Translation, Seq2Seq and Attention**  [[slides](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1194/slides/cs224n-2019-lecture08-nmt.pdf)] [[video](https://youtu.be/XXtpJxZBa2c)] [[notes](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1194/readings/cs224n-2019-notes06-NMT_seq2seq_attention.pdf)] [[assignment](https://github.com/VinAIResearch/DL4NLP/blob/master/Assignment/Sentiment%20analysis/Assignment%202%20-%20attention%20LSTM.txt)]

	Suggested Readings:

	-  [Statistical Machine Translation slides, CS224n 2015](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1162/syllabus.shtml)  (lectures 2/3/4)
	-  [Statistical Machine Translation](https://www.cambridge.org/core/books/statistical-machine-translation/94EADF9F680558E13BE759997553CDE5)  (book by Philipp Koehn)
	-  [BLEU](https://www.aclweb.org/anthology/P02-1040.pdf)  (original paper)
	-  [Sequence to Sequence Learning with Neural Networks](https://arxiv.org/pdf/1409.3215.pdf) (original seq2seq NMT paper)
	-  [Sequence Transduction with Recurrent Neural Networks](https://arxiv.org/pdf/1211.3711.pdf)  (early seq2seq speech recognition paper)
	-  [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/pdf/1409.0473.pdf)  (original seq2seq+attention paper)
	-  [Attention and Augmented Recurrent Neural Networks](https://distill.pub/2016/augmented-rnns/)  (blog post overview)
	-  [Massive Exploration of Neural Machine Translation Architectures](https://arxiv.org/pdf/1703.03906.pdf)  (practical advice for hyperparameter choices)

- **26/02/2021: Question Answering** [[slides](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1194/slides/cs224n-2019-lecture10-QA.pdf)] [[video](https://youtu.be/yIdF-17HwSk)] [[notes](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1194/readings/cs224n-2019-notes07-QA.pdf)]
- **05/03/2021: ConvNets for NLP**  [[slides](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1194/slides/cs224n-2019-lecture11-convnets.pdf)] [[video](https://youtu.be/EAJoRA0KX7I)] [[notes](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1194/readings/cs224n-2019-notes08-CNN.pdf)] [[assignment](https://github.com/VinAIResearch/DL4NLP/blob/master/Assignment/Sentiment%20analysis/Assignment%203%20-%20CNN.txt)]

	Suggested Readings:

	-  [Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882)
	-  [A Convolutional Neural Network for Modelling Sentences](https://arxiv.org/pdf/1404.2188.pdf)

- **12/03/2021: Information from parts of words: Subword Models**  [[slides](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1194/slides/cs224n-2019-lecture12-subwords.pdf)] [[video](https://youtu.be/9oTHFx0Gg3Q)]

	Suggested reading:  [Achieving Open Vocabulary Neural Machine Translation with Hybrid Word-Character Models](https://arxiv.org/abs/1604.00788)

- **19/03/2021:** 
    - **Modeling contexts of use: Contextual Representations and Pretraining**  [[slides](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1194/slides/cs224n-2019-lecture13-contextual-representations.pdf)] [[video](https://youtu.be/S-CspeZ8FHc)] [[assignment](https://github.com/VinAIResearch/DL4NLP/blob/master/Assignment/Sentiment%20analysis/Assignment%204%20-%20BERT.txt)]
    - **PhoBERT: Pre-trained language models for Vietnamese** [[Slides]](https://datquocnguyen.github.io/resources/DatQuocNguyen_AIDay2020_FinalVersion.pdf) [[video]](https://youtu.be/dqZitP00xXw?t=4092) [[assignment]](https://github.com/VinAIResearch/DL4NLP/tree/master/Assignment/Sequence_Labeling)
    
	Suggested readings:

	1.  [Contextual Word Representations: A Contextual Introduction](https://arxiv.org/abs/1902.06006) 
	2.  [The Illustrated BERT, ELMo, and co](http://jalammar.github.io/illustrated-bert/)
	
- **Natural Language Generation** (Optional) [[slides](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1194/slides/cs224n-2019-lecture15-nlg.pdf)] [[video](https://youtu.be/4uG1NMKNWCU)] 
- **Reference in Language and Coreference Resolution** (Optional)  [[slides](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1194/slides/cs224n-2019-lecture16-coref.pdf)] [[video](https://youtu.be/i19m4GzBhfc)]
- **Multitask Learning: A general model for NLP?** (Optional) [[slides](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1194/slides/cs224n-2019-lecture17-multitask.pdf)] [[video](https://youtu.be/M8dsZsEtEsg)]

### Additional resource

- [DEEP LEARNING FOR NLP WITH PYTORCH](https://pytorch.org/tutorials/beginner/deep_learning_nlp_tutorial.html)
