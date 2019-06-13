# Technical Report - Quora Sincere Question Classification
### By Egan Bailey, SF DSI 7

# Problem Statement
Natural Language Processing (NLP) is an important subfield of data science with a variety of applications. [Quora](https://www.quora.com/) is a question-and-answer website, where users submit questions that they want answered, and other community members answer those questions. These questions are 

Quora has an issue with “insincere” questions. These include questions that violate their terms of service, or the spirit of those rules. Oftentimes, questions are “asked” in the form of a statement, not a question. The question may contain profanity, or be framed in such a way as to be non-neutral or even insulting. Such questions hurt user experiences, and enflame subsequent discourse. Quora can’t individually curate their questions, and so must come up with a model that can help them identify insincere questions.

The goal of this project is to take the giant dataset and find a way to effectively classify questions as "sincere" or "insincere". The metric that we will be using will be the same as the metric of the Kaggle competition: F1-score. The higher the F1 score, the better.

Our datasets are giant, 1.3 million in the training and ~375,000 in the test. Each iteration of my code is a slog as my Windows desktop's GPU tries valiantly to run Keras Tensorflow models on a LSTM Recurrent Neural Network (RNN). I had big dreams for this dataset. More epochs, more embeddings, more layers. But ultimately I couldn't get my AWS server to cooperate with Jupyter Notebooks, so I haven't been able to take advantage of some of AWS's computer power.

# Data Collection
I used a dataset from Kaggle that was curated specifically for this competition by Quora. The data consists of a training set of questions and a target of "sincere" or "INsincere",  random sample of review, user, and business data for academic or research use. The dataset can be found [here](https://www.kaggle.com/c/quora-insincere-questions-classification/data).

The data is helpfully in .csv format. In addition, there are a series of suggested embeddings (FastText gave my PC a ton of issues, but that's one that I'd like to try to meld together in a model). Initially, my laptop was up to the task of running the code. Eventually, however, once it got to modelling and predicting-time, I ran into an issue because I didn't have a Keras-recognizeable GPU. Trying to only run on CPU was a nightmare, so I turned to my PC, which ran the models and the kaggle kernels.

# Exploratory Data Analysis

The training dataset contained ~1.3 million questions from a variety of users in Quora.  

The distribution of useful votes is described in the histogram below:

![Useful dist](https://git.generalassemb.ly/eganbailey/Egan-Bailey-Capstone-DSI-7/blob/master/Egan%20Bailey%20Capstone%20Code/Pic%201.png)
---
## Word Cloud

The word cloud seems to indicate some interesting trends that seem to make sense. As the words become more divisive or political, they tend to appear in more insincere questions.

Sincere Word Cloud
![Useful cloud](https://git.generalassemb.ly/eganbailey/Egan-Bailey-Capstone-DSI-7/blob/master/Egan%20Bailey%20Capstone%20Code/Pic%202.png)

Insincere Word Cloud
![Useful Cloud Insincere](https://git.generalassemb.ly/eganbailey/Egan-Bailey-Capstone-DSI-7/blob/master/Egan%20Bailey%20Capstone%20Code/Pic%203.png)

# Modeling: Neural Networs: GRU and LSTM Models

I created a pair of neural networks using multiple layers of embeddings.

For the LSTM model, I used both the GloVe and Paragrams embeddings, along with tokenization and padding, to create a model that ran a series of 4 randomly-sliced epoch sets of 8 epochs each. These epochs were randomly tweaked by another layer called the "attention" layer, which I adapted from this source [here](https://www.kaggle.com/suicaokhoailang/lstm-attention-baseline-0-652-lb).

I did some research into different neural network models, and stumbled upon GRU's (Grated Recurremt Unit). In this model, I tried using 3 different embeddings, all tweaked by a "capsule", in order to create a GRU model that would be faster than my LSTM model. Unfortunately the end result crashed my computer multiple times, so I shrunk the dataset in half. My end result, only 0.7% worse than the LSTM model, makes me think that I might be able to get a better score rather easily simply by running my GRU model to it's fullest potential.

---
## Results

I first tried my hand at a series of unsupervised models devoid of embeddings. These, as my presentation makes abundantly clear, were massively ineffective. A table of their results is below:

|          | GS K-NN | GS Bagging | GS Trees | Naïve Bayes |
|----------|---------|------------|----------|------------:|
| Accuracy | .93     | .944       | .944     | .948        |
| F1 Score | N/A     | .0625      | .0625    | .066        |

These results are actually pretty awful, because they're not very far off from a very unbalanced training and test set, which contains sincere questions at a rate of 93.8 percent.

For the businesses data, an LSTM network, using two simultaneous layers of embeddings (GloVe and Paragram), was able to create a neural network that classifies Quora questions as sincere or insincere with an F1 score of 68.8%, the metric of choice for the Kaggle Kernel competition. This is superior to my GRU network, which produced an F1 score of 68.1%.

This result is actually about 13.7% better than my previous score of 60.54 percent, which is a significant jump despite low computer resources. I attempted to create more complex models with more layers, nodes, or using 3-4 embeddings in the LSTM model, but my kernels crashed almlost instantly, and my computers simply could not hope to keep up. 

# Future Steps

**Improve Language Recognition** - Dataset includes non-English words/characters (e.g asking a question related to the meaning of a foreign language). A program that allows for the translation and understanding of foreing languages may result in a useful weighting to any neural network, as I believe that many questions containing foreign lanaguages are going to be "sincere".

**More Computer Power** - Better solutions involve using far greater amounts of computing power, likely supplied by AWS. My computer spends between 3-4 hours each time I try to run my model fully, which is practically lightening speed compared to the cursed Word2Vec model that I created in my first iteration of this project.
