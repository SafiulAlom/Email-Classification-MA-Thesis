#Diese Datei importiert alle notwendige Paketen und Dateien

import datetime
import pandas as pd
import numpy as np
import gzip
import sys
import joblib
import matplotlib.pyplot as plt
#%matplotlib inline
from nltk.corpus import stopwords
import features.zedsets
import seaborn as sns
import spacy
import textwrap
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn import preprocessing, model_selection
from sklearn.metrics import accuracy_score, confusion_matrix, \
recall_score, roc_auc_score, precision_score, f1_score, log_loss,\
roc_curve, auc
import sklearn.metrics as sk
from sklearn.utils import shuffle
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelBinarizer,label_binarize
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, RepeatedStratifiedKFold
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import GermanStemmer
import lightgbm
import spacy
import re
import multiprocessing
from scipy import sparse
nlp = spacy.load('de_core_news_sm')
from inspect import getsource
import pickle
import sys
import scipy
import scipy.sparse as sp
from gensim.models.fasttext import load_facebook_model, smart_open
import gensim
from gensim.sklearn_api import TfIdfTransformer, LsiTransformer
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import Dense, Dropout, Activation,\
Flatten, GlobalAveragePooling1D, Embedding, GlobalMaxPool1D, LSTM,\
Input, LSTM,Bidirectional,Activation,Conv1D,GRU,\
MaxPooling1D, AveragePooling1D, Add, concatenate, \
SpatialDropout1D, Layer, Bidirectional 
from tensorflow.keras import Sequential, initializers, Model, activations
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd

