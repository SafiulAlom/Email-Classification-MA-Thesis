#Diese Datei listet alle Parameter-Werte, 
#die zur Erstellung der ML/DL-Modelle verwendet werden
import os
os.chdir('/home/safiul/email-classification')
import Email_Classification.Functions.Functions as fn
from Email_Classification.Functions.packages import *

param_process = dict(
    sample_size = 3105360,
    test_size = 0.1,
    nchar = 2000,
    rmDigits = True,
    trans_lower = True,
    stem_lemma = 'stem',
    num_word = 1500,
    maxlen = 2000,
    embedding_dim = 300,
    pooling = 'max',
    n_split = 200
)

param_nlp = dict(
    no_below = 1,
    no_above = 1.0,
    keep_n = 50000,
    num_topics = 2000,
    onepass = True,
    power_iters = 5,
    ngram_range = (1,1), 
    max_features= 2000
)

param_nb = dict(
    alpha= 0.01, 
    fit_prior=True
)

param_lgb = dict(
    parameters = dict(
    objective = 'multiclass',
    num_class = 100,
    is_unbalance = 'true',
    boosting =  'gbdt',
    num_leaves = 50,
    feature_fraction =  0.5,
    feature_fraction_seed = 128,
    bagging_fraction =  0.5,
    bagging_fraction_seed =  123,
    bagging_freq = 20,
    learning_rate = 0.06,
    lambda_l2 = 1.0,
    seed= 147
        ),
    num_boost_round= 1000,
    early_stopping_rounds= 20,
    feval= fn.accuracy_validate
)

param_rf = dict(
    random_state=5,
    criterion='entropy',
    max_features= 'sqrt',
    n_estimators = 128,
    min_samples_split = 50,
    max_samples = 1000000,
    min_impurity_decrease = 1e-5,
    max_depth = 300,
    n_jobs = -1,
    verbose= 2
)

param_lr = dict(penalty = 'l2',
                max_iter= 200,
                C = 1.0,
                fit_intercept = False,
                tol = 1e-4,
                solver = 'sag',
                multi_class='multinomial',
                n_jobs = -1,
                verbose = 1,
                random_state = 125
)

param_mlp = dict(seed = 1234,
                 nb_units = 128,
                 learning_rate=0.001, 
                 beta_1=0.9, 
                 beta_2=0.999, 
                 epsilon=1e-07,
                 batch_size = 32, #1 for online SGD
                 epochs = 30,
                 verbose = 1,
                 n_h = [128, 128],
                 dropout_prob = 0.5,
                 training = False               
)



param_embed = dict(input_dim = 50000, 
                   output_dim = 300, 
                   trainable = False,
                   quantile = 1.0
)

param_conv = dict(filters = 200, 
                  kernel_size = 5, 
                  padding="valid", 
                  activation_conv = "relu", 
                  strides=1,
                  use_bias=True,
                  kernel_initializer = tf.keras.initializers.he_normal(seed = 1234),
                  bias_initializer="zeros",
                  pool_size = 4,
                  rate_spatialDropout = 0.5,
                  dropout = 0.5
)

param_lstm = dict(units = 200,
                  activation="tanh",
                  recurrent_activation="sigmoid",
                  use_bias=True,
                  kernel_initializer = tf.keras.initializers.he_normal(seed = 1234),
                  recurrent_initializer= tf.keras.initializers.Orthogonal(seed= 1234),
                  bias_initializer="zeros",
                  dropout = 0.5,
                  recurrent_dropout = 0.5,
                  return_sequences=True
)


prior_params = dict(prior_sigma_1 =  1.5, 
                    prior_sigma_2 = 0.1, 
                    prior_pi = 0.5,
                    prior = 'gaussian'
)

param_optimizer = dict(learning_rate=0.001, 
                       beta_1=0.9, 
                       beta_2=0.999, 
                       epsilon=1e-07, 
                       name='Adam'
)

param_fit = dict(wkd = 'Email_Classification/Models/Sequential Models/rnn/lstm/Checkpoint/',
                      monitor='val_accuracy', 
                      patience= 5, 
                      min_delta= 1e-3, 
                      mode='max',
                      save_best_only= True,
                      verbose= 1,
                      batch_size= 128,
                      epochs= 30
)

params = dict(param_process = param_process,
              param_nlp = param_nlp,
              param_nb = param_nb,
              param_lgb = param_lgb,
              param_rf = param_rf,
              param_lr = param_lr,
              param_mlp = param_mlp,
              param_embed = param_embed,
              param_conv = param_conv,
              param_lstm = param_lstm,
              prior_params = prior_params,
              param_optimizer = param_optimizer,
              param_fit = param_fit
)
