## python-sparselda
================
python-sparselda is a Latent Dirichlet Allocation topic modeling package based on SparseLDA Gibbs Sampling inference algorithm, and developed with Python.

The source code package will be coming soon.


Please use the github issue tracker for python-sparselda at:
https://github.com/fandywang/python-sparselda/issues

## Usage
================
### 1. Install Google Protocol Buffers (protobuf)
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
Help:

    python lda_trainer.py -h
    
Example:

    python lda_trainer.py \
    --corpus_dir=testdata/corpus/ \
    --vocabulary_file=testdata/vocabulary.dat \
    --num_topics=10 \
    --topic_prior=0.1 \
    --word_prior=0.01 \
    --save_model_interval=100 \
    --model_dir=testdata/sparselda_models \
    --is_save_checkpoint=True \
    --save_checkpoint_interval=10 \
    --checkpoint_dir=testdata/sparselda_checkpoints
    
### 3. Inference
### 4. Evaluation

## References
================
1. D. Blei, A. Ng, and M. Jordan. Latent Dirichlet allocation. Journal of Machine Learning Research, 2003.
2. Gregor Heinrich. Parameter estimation for text analysis. Technical Note, 2004.
3. Griﬃths, T. L., & Steyvers, M. Finding scientiﬁc topics. Proceedings of the National Academy of Sciences(PNAS), 2004.
4. I. Porteous, D. Newman, A. Ihler, A. Asuncion, P. Smyth, and M. Welling. Fast collapsed Gibbs sampling for latent Dirichlet allocation. In SIGKDD, 2008.
5. Limin Yao, David Mimno, Andrew McCallum. Efficient methods for topic model inference on streaming document collections, In SIGKDD, 2009.
7. X. Wei, W. Bruce Croft. LDA-based document models for ad hoc retrieval. In Proc. SIGIR. 2006.
8. Rickjin, LDA 数学八卦. Technical Note, 2013.
