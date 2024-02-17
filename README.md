# Instagram Comment Sentiment Analysis with BERT

## Project Overview

This tool fine-tunes a BERT Sentiment Analyzer to assign positive, negative, or neutral sentiment to a set of Instagram comments.
This version of the model is scoped to English-language comments, and it ignores URLs and emojis.

## Installation

This repository uses [pipenv](https://pipenv.pypa.io/en/latest/installation.html) for package management.

Clone the repository and install everything using pipenv:

```bash
git clone https://github.com/cgoods94/instagram-comment-sentiment.git
cd instagram-comment-sentiment
pipenv shell
pipenv install
```

## Setup

To get the main.py file up and running, you'll need to add the following to your cloned repository.

### 1. Environment (.env) File

In the root directory of this repository, add a `.env` file with the following inside:

```bash
INSTAGRAM_USER=<your_instagram_username>
INSTAGRAM_PASS=<your_instagram_password>
```

You'll want to exit pipenv and re-enter with `pipenv shell` to see the .env changes take place.

### 2. Preparing For and Training the Model and Tokenizer

The trained BERT model and tokenizer are rather large for a personal Github project. As a solo user without enterprise resources, I have not included them in the remote repository. However, I've built training set functionality that takes a .xlsx file (some Unicode was causing issues in .csv) of Instagram comment text with labels, and I have included a small .xlsx training set as an example.

#### a. Create Model and Tokenizer Directories

Create two folders in the `model` directory called `models` and `tokenizers`

From the root directory of this repository:

```bash
mkdir src/python/model/models
mkdir src/python/model/tokenizers
```

#### b. Build & Train the BERT Model and Tokenizer

I have provided a .xlsx file with the labeling conventions that the model training expects.

From a file like that, you can run `train_example_model.py` from the `src/python` directory, which will 
save a trained model and its tokenizer to the respective filepaths from the previous step.

Even with a small training set like the given set, it takes some time to train on a CPU (30-40 min on a 2017 MBP, for example).

## Usage

I've provided an example file that uses the model on a set of Instagram comments taken from an
ESPN post about Taylor Swift flying her private jet from Japan to the Super Bowl.

You can run `src/python/main.py` for yourself to see.

## Limitations / Plans for Improvement

This stage is a first pass on the concept, and there are several things I'm working to improve.

### 1. Does Not Handle Emojis

While I plan on sticking to the English-language and non-URL scope of this project, I would like
to extract sentiment from emojis somehow.

### 2. Small Training Set

I hand-labeled ~1,000 Instagram comments on my couch, which only gives the model enough to reach
~72% validation accuracy after 4 epochs (30-40 min of training/validation time). I'm planning
on exploring semi-supervised training to see about augmenting this training set.

### 3. Instagram API Limits

There is some code in the `training` folder related to downloading Instagram comments at scale.
However, the API does limit requests to 200/hr. This seriously limits training data collection time. 
In addition, I made a dev account on Instagram to curate a following list that would yield a good
training set. Instagram does not love accounts created for purposes like this, as a heads up.

### 4. Slow Training Time

On my MacBook Pro, this takes 30-40 min to train on the 1k examples I have. The code is set up for
GPU usage, but I do not currently have access to a GPU. I'd be open to exploring affordable options there.

## Contact

If you have any questions or want to connect with me on other data science projects/opportunities,
you can reach me at cgoods94@gmail.com