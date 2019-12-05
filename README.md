# Text-Analytics-Language-Modeling

**Author**: Saurabh Annadate

In this work, we explore three model architectures which include statistical N-gram models and two Recurrent Neural Network architectures for large scale language modelling. A subset of the Gutenberg eBooks corpora was used as the corpus on which the models are fit.

## Dataset
The Project Gutenberg corpus (http://www.gutenberg.org/) was considered for our analysis. Project Gutenberg is a library of over 60,000 free eBooks. The books in the project repository have been chronologically assigned a serial number which goes from 1 to ~62000. Once the books are downloaded, the following operations are performed to create the final corpus:
1. Books are filtered for English books
2. Book metadata and the Gutenberg license is removed from all the books’ texts
3. All books’ texts are concatenated
4. The final corpus is cleaned by removing the occurrences of unwanted characters like ‘\n’ and multiple spaces between words
5. The sentences were padded with begin and end of sentence tokens for the N-gram modelling
After creating the final corpus, the text is tokenized using NLTK word_tokenize and sent_tokenize

## Models
The following three models were fit on the data for language modelling:

### N-Gram models
We trained four N-Gram models for N = 2, 3, 4 and 5 using NLTK lm module. This basic module does not perform any smoothing and hence outputs the probabilities for unseen sequences as zero. As a result, we were unable to calculate the perplexity metric which is generally used to report performance of language models. 

### Character level Neural Net
Figure 1 depicts the model architecture used for character level neural net model. The input is a sequence of 128 characters and the model aims to predict the next character. The model consists of a Keras embedding layer which creates embeddings of dimension 128 for the input characters. This is followed by two or more recurrent neural network layers with varying hidden states. The final layer is a softmax dense layer over the entire character set which calculates the probability of the next character. There are 157 unique characters in our training set over which the model calculates the probability. Different model architectures are constructed by varying the number of hidden states of the RNN, number of RNN layers as well as type of RNN unit (LSTM or GRU). The loss that is minimized is categorical cross-entropy. All the models are trained for 5 epochs and the training time and validation cross entropy is recorded.

### Word level Neural Net
The biggest challenge with constructing a word level neural net is that the final softmax layer needs to predict over the entire vocab which may contain millions of words and hence becomes computationally intensive. Although there are techniques to count this like Noise Contrastive Estimation (NCE) loss, self-normalizing partition functions, or hierarchical softmax, we decided to adopt a slightly different approach. Instead of modelling the probability of the next word, we modelled the word embeddings. Figure 2 shows the model architecture. First, a word2vec model is fit on the entire training corpus. A sequence of 128 words is passed through the word2vec model to get the embeddings which form the input for the neural net model. This is followed by two or more recurrent neural network layers with varying hidden states. The final layer is a tanh dense layer of dimension 300 which predicts the embedding of the next word. The loss which is optimized for is mean square error. Different model architectures are constructed by varying the number of hidden states of the RNN, number of RNN layers as well as type of RNN unit (LSTM or GRU). All the models are trained for 5 epochs and the training time and validation mse is recorded.

## How to use the Repo

#### Specifying configurations
All configurations are specified in the **Config/config.yml** file. The file structure is as follows:

```
logging:
  LOGGER_NAME: 'root'

fetch_data:
  indices:
    start: 45000
    end: 62000
  save_location: "Data/raw/"

clean_data:
  save_location: "Data/clean/"

create_corpus:
  save_location: "Data/processed/"

gen_training:
  char_nn_training_size: 2000000
  # This is the total character size that you would like to take for training
  # Put -1 if you do not want to limit the number of characters to be taken from the corpus to train
  word_nn_training_size: 2000000

n_gram:
  gram_count: 2
  model_name: "two_model"
  validation_split: 0.1

char_nn:
  seq_length: 128
  batch_size: 250
  embedding_dim: 128
  rnn_type: "lstm"   #can be "lstm" or "gru"
  rnn_layers: 2
  rnn_units: 256
  dropout: 0.3
  epochs: 2
  validation_split: 0.1
  l2_penalty: 0.0003
  model_name: "char_neural_model1"

w2v_model:
  model_name: "word2vec_300_model1"
  size: 300
  min_count: 100
  workers: 4
  window: 10
  sg: 1 #1 or 0
  hs: 1 #1 or 0

word_nn: 
  w2v_model: "Models/word2vec/word2vec_300_model1.model"
  seq_length: 128
  batch_size: 250
  embedding_dim: 300
  rnn_type: "gru"   #can be "lstm" or "gru"
  rnn_layers: 5
  rnn_units: 256
  dropout: 0.3
  epochs: 5
  validation_split: 0.1
  l2_penalty: 0.0003
  model_name: "word_neural_model6"
  
char_api:
  model_name: "char_neural_model3"
  seq_length: 128

word_api:
  w2v_model: "Models/word2vec/word2vec_300_model1.model"
  embd_size: 300
  model_name: "word_neural_model1"
  seq_length: 128
  
ngram_api:
  model_name: "three_model"
```

Edit this file in this file to experiment with different configurations.

### Downloading the corpus
```
python run.py fetch_data
```
The serial numbers of the books to fetch are specified in the config file.
<br><br>
### Cleaning and creating corpus
```
python run.py clean_data
python run.py create_corpus [--docCount=500]
```
--docCount (Optional; default = 500): Number of documents to be taken to create the corpus
<br><br>
### Fit word2vec model
```
python run.py run_w2v
```
<br><br>
### Fit character level neural net model
```
python run.py run_char_nn
```
<br><br>
### Fit word level neural net model
```
python run.py run_word_nn
```
<br><br>
### Fit N-Gram model
```
python run.py run_ngram
```
<br><br>
### Launch character level neural net model API
```
python run.py run_char_api
```
<br><br>
### Launch word level neural net model API
```
python run.py run_word_api
```
<br><br>
### Launch N-Gram model api
```
python run.py run_ngram_api
```




