## python-sparselda
================
python-sparselda is a Latent Dirichlet Allocation topic modeling package based on SparseLDA Gibbs Sampling inference algorithm, and written in Python 2.6 or newer, Python 3.0 or newer excluded. 

Frankly, python-sparselda is just a mini project, we hope it can help you better understand the standard LDA and SparseLDA algorithms. RTFSC for more details. Have fun.

Please use the github issue tracker for python-sparselda at:
https://github.com/fandywang/python-sparselda/issues

## Usage
================
### 1. Install Google Protocol Buffers
python-sparselda serialize and persistent store the lda model and checkpoint based on protobuf, so you should install it first.

    wget https://protobuf.googlecode.com/files/protobuf-2.5.0.tar.bz2
    tar -zxvf protobuf-2.5.0.tar.bz2
    cd protobuf-2.5.0
    ./configure
    make
    sudo make install
    cd python
    python ./setup.py build
    sudo python ./setup.py install

    cd python-sparselda/common
    protoc -I=. --python_out=. lda.proto

### 2. Training
    Usage: python lda_trainer.py [options].

    Options:
    -h, --help   show this help message and exit
    --corpus_dir=CORPUS_DIR
            the corpus directory.
    --vocabulary_file=VOCABULARY_FILE
            the vocabulary file.
    --num_topics=NUM_TOPICS
            the num of topics.
    --topic_prior=TOPIC_PRIOR
            the topic prior alpha.
    --word_prior=WORD_PRIOR
            the word prior beta.
    --total_iterations=TOTAL_ITERATIONS
            the total iteration.
    --model_dir=MODEL_DIR
            the model directory.
    --save_model_interval=SAVE_MODEL_INTERVAL
            the interval to save lda model.
    --topic_word_accumulated_prob_threshold=TOPIC_WORD_ACCUMULATED_PROB_THRESHOLD
            the accumulated_prob_threshold of topic top words.
    --save_checkpoint_interval=SAVE_CHECKPOINT_INTERVAL
            the interval to save checkpoint.
    --checkpoint_dir=CHECKPOINT_DIR
            the checkpoint directory.
    --compute_loglikelihood_interval=COMPUTE_LOGLIKELIHOOD_INTERVAL
            the interval to compute loglikelihood.
### 3. Inference
Please refer the example: lda_inferencer.py. Note that we strongly recommend you to use MultiChainGibbsSampler class.

### 4. Evaluation
Instead of manual evaluation, we want to evaluate topics quality automatically, and filter out a few meaningless topics to enchance the inference effect.

## TODO
================
1. Hyperparameters optimization.
2. Memory optimization.
3. More experiments.
4. Data and model parallelization.

## References
================
1. Blei, A. Ng, and M. Jordan. Latent Dirichlet allocation. Journal of Machine Learning Research, 2003.
2. Gregor Heinrich. Parameter estimation for text analysis. Technical Note, 2004.
3. Griﬃths, T. L., & Steyvers, M. Finding scientiﬁc topics. Proceedings of the National Academy of Sciences(PNAS), 2004.
4. I. Porteous, D. Newman, A. Ihler, A. Asuncion, P. Smyth, and M. Welling. Fast collapsed Gibbs sampling for latent Dirichlet allocation. In SIGKDD, 2008.
5. Limin Yao, David Mimno, Andrew McCallum. Efficient methods for topic model inference on streaming document collections, In SIGKDD, 2009.
6. Newman et al. Distributed Inference for Latent Dirichlet Allocation, NIPS 2007.
7. X. Wei, W. Bruce Croft. LDA-based document models for ad hoc retrieval. In Proc. SIGIR. 2006.
7. Rickjin, LDA 数学八卦. Technical Note, 2013.
8. Yi Wang, Hongjie Bai, Matt Stanton, Wen-Yen Chen, and Edward Y. Chang. PLDA: Parallel Latent Dirichlet Allocation for Large-scale Applications. AAIM 2009.

## Links
===============
Here are some pointers to other implementations of LDA.

1. [LDA-C](http://www.cs.princeton.edu/~blei/lda-c/index.html): A C implementation of variational EM for latent Dirichlet allocation (LDA), a topic model for text or other discrete data.
2. [GibbsLDA++](http://gibbslda.sourceforge.net/): A C/C++ implementation of Latent Dirichlet Allocation (LDA) using Gibbs Sampling technique for parameter estimation and inference.
3. [Matlab Topic Modeling Toolbox](http://psiexp.ss.uci.edu/research/programs_data/toolbox.htm)
4. [lda-j](http://www.arbylon.net/projects/): Java version of LDA-C and a short Java version of Gibbs Sampling for LDA.
5. [Mr. LDA](https://github.com/lintool/Mr.LDA): A Latent Dirichlet Allocation topic modeling package based on Variational Bayesian learning approach using MapReduce and Hadoop, developed by a Cloud Computing Research Team in University of Maryland, College Park.
6. [Yahoo_LDA](https://github.com/sudar/Yahoo_LDA): Y!LDA Topic Modelling Framework, it provides a fast C++ implementation of the inferencing algorithm which can use both multi-core parallelism and multi-machine parallelism using a hadoop cluster. It can infer about a thousand topics on a million document corpus while running for a thousand iterations on an eight core machine in one day.
7. [plda/plda+](https://code.google.com/p/plda/): A parallel C++ implementation of Latent Dirichlet Allocation (LDA).
8. [Mahout](https://cwiki.apache.org/confluence/display/MAHOUT/Latent+Dirichlet+Allocation): Mahout's goal is to build scalable machine learning libraries. 
9. [MALLET ](http://mallet.cs.umass.edu/): A Java-based package for statistical natural language processing, document classification, clustering, topic modeling, information extraction, and other machine learning applications to text.
10. [peacock](): 
