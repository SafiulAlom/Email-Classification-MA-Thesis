#Dieses Datei enthält alle Funktionen zur E-Mail-Klassifikation

import os
#Arbeitsverzeichnis festlegen
os.chdir('/home/safiul/email-classification')
#Skripte der aufgelisteten Pakete laden
from Email_Classification.Functions.packages import *

#listet Top-n-Wörter aus allen E-Mail-Korpus auf
def get_top_n_words(corpus, stopWords = None, n = None):
    vec = CountVectorizer(stop_words = stopWords).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]


#Erstellt eine Tabelle mit nötwendigen Spalten aus den gegebenen Tabellen
#Es werden der Betreff und Body der E-Mails zusammengefügt
#Zusätzlich werden neue Spalten mit den Informationan der E-Mail-Länge und 
#der anzahl der Wörter  in den E-Mails eingefügt
def createDataFrameEmail(headers, bodies, targets, sample_size, seed):
    targets = targets.query('is_target_category_rest_class == False')
    headers_bodies = pd.merge(headers, bodies[['body', 'email_id']], on = 'email_id')
    headers_bodies_targets = pd.merge(headers_bodies, targets[['target_category', 'email_id']], on = 'email_id')
    emails = headers_bodies_targets[['subject', 'body', 'target_category']]
    if sample_size > emails.shape[0]:
        sample_size = emails.shape[0]
    emails = emails.sample(sample_size, random_state = seed)
    emails = emails.replace(to_replace = np.NaN, value = '')
    emails['emails'] = emails['subject'] + ' ' + emails['body']
    emails['emails'] = emails['emails'].str.strip()
    emails = emails[emails.emails != ''].copy()
    emails['emails_len'] = emails.emails.astype(str).apply(len)
    emails['word_count'] = emails.emails.apply(lambda x: len(str(x).split()))
    return emails


#Verarbeitet den E-Mail-Datensatz zur Klassifikation vor (Phase 1)
#wandelt die Texte in den kleinen Buchstaben um
#schneidet die E-Mails nach einer bestimmten Stelle ab
#entfernt unnötige Bezeichnungen
class Text_process(BaseEstimator, ClassifierMixin):
    
    def __init__(self, nchar = 1000, rmDigits = True, trans_lower=True):
        self.nchar = nchar
        self.rmDigits = rmDigits
        self.trans_lower = trans_lower
    
    def transform_lower(self, text):
        if self.trans_lower:
            text = text.lower()
        return text
    
    def CutOff(self, text):
        return textwrap.wrap(text, self.nchar)[0]
    
    def rm_text_byPattern(self, pattern, text):
        return re.sub(pattern,' ', text, flags= re.MULTILINE) if re.findall(pattern, text) else text
    
    def rm_irrelvant_text(self, text):
        
        text = self.transform_lower(self.CutOff(text))
        
        if self.rmDigits: 
            patterns = [r'\\t|\\n|\\r', r'\t|\n|\r', r'\W', r'\b\w\b', '[\d+_]', '\s+', r'xxx']
        else:
            patterns = [r'\\t|\\n|\\r', r'\t|\n|\r', r'\W', r'\b\w\b', '\s+', r'xxx']
            
        if len(text.strip().split()) < 2:
            text = text.strip()
        else:
            for pattern in patterns:
                text = self.rm_text_byPattern(pattern, text)
                text = ' '.join( [w for w in text.split() if len(w)>1] )
        return text
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y = None):
        return X.copy().apply(self.rm_irrelvant_text)
    
    def fit_transform(self, X, y = None):
        self.fit(X, y)
        return self.transform(X, y)
 
     
#Verarbeitet den E-Mail-Datensatz zur Klassifikation vor (Phase 2)
#Entfernt die von nltk vorgeschlagenen  sowie Domäne-abhängigen Stoppwörter
#Führt Lemmatisierung oder Stemming durch
class Text_normalizer(BaseEstimator, ClassifierMixin):
    
    def __init__(self, stem_lemma = 'stem'):
        #self.stopword = stopword
        self.stem_lemma = stem_lemma
    
    def rm_stopwords(self, email):
        stopword = set(stopwords.words('german'))
        stopword = list(stopword.union({'re', 'aw', 'dkb', 'ag', 'www', '2019',\
                                       'damen', 'herren', 'freundlichen', 'deutsche', 'kreditbank',\
                                       'geehrte'}))
        return [word for word in word_tokenize(email) if not word in stopword]
    
    def stemming(self, email):
        stemmer = GermanStemmer()
        email = self.rm_stopwords(email)
        email = [stemmer.stem(token) for token in email]
        return ' '.join(email)
    
    def lemmatization(self, email):
        email = self.rm_stopwords(email)
        email = nlp(' '.join(email))
        email = [token.lemma_ for token in email]
        return' '.join(email)
    
    def choose_stem_lemma(self, email):
        if(self.stem_lemma == "stem"):
            email =  self.stemming(email)
        if(self.stem_lemma == "lemma"):
            email = self.lemmatization(email)
        return email
         
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y = None):
        return X.copy().apply(self.choose_stem_lemma)
    
    def fit_transform(self, X, y = None):
        self.fit(X, y)
        return self.transform(X, y)
    

# Parallelisiert die Berechnungen    
def multiprocess_array(Model, X):
    n_jobs = multiprocessing.cpu_count()
    X_split = np.array_split(X, n_jobs)
    with multiprocessing.Pool(n_jobs) as pool:
        X_processed = list(
                pool.map(
                    Model.transform,
                    X_split
                )
            ) 
    return X_processed   


#Schreibt die ML-Modelle in die Festplatte       
def writemodel(path, filename, variable, makeZip):
    if makeZip == True:
        with gzip.open(os.path.join(path, filename + '.pklz'), 'wb') as file:
            pickle.dump(variable, file, protocol=-1) 
    else:
        with open(os.path.join(path, filename + '.pkl'), 'wb') as file:
            pickle.dump(variable, file, protocol=-1) 
            
        
#lädt die ML-Modelle aus der Festplatte  zum Weiterverarbeiten    
def readmodel(path, makeZip):
    listfiles = os.listdir(path)
    listfiles = [f for f in listfiles if not f.startswith('.')]
    dict_model = {}
    if makeZip == True:
        for f in listfiles:
            with gzip.open(os.path.join(path, f), 'rb') as file:
                dict_model[os.path.splitext(f)[0]] =  pickle.load(file)
    else:
        for f in listfiles:
            with open(os.path.join(path, f), 'rb') as file:
                dict_model[os.path.splitext(f)[0]] =  pickle.load(file)   
    return dict_model       


#Benutzer definierte Funktion zur Berechnung der Genauigkeit
def accuracy_validate(y_pred, train_data):
    labels = train_data.get_label().astype("int")
    y_pred = y_pred.reshape(len(np.unique(labels)), -1).argmax(axis=0)
    accuracy = accuracy_score(labels, y_pred)
    return 'accuracy', accuracy, True


#Erweiterte Vorverarbeitung zum Word-Embedding
#tokenisiert jede E-Mail in einzelne Wörter
#führt Padding durch
#erstellt die Embedding-Matrix mithilfe einer vortrainierten FastText-Modell
#erzeugt 1-D Vektoren unter Verwendung von Average oder Maximum-Pooling
class Embedding_custom(BaseEstimator, ClassifierMixin):
    def __init__(self, 
                 max_feature,
                 X_train,
                 X_test, 
                 quantile):
        self.max_feature = max_feature
        self.X_train = X_train
        self.X_test = X_test
        self.quantile = quantile
    
    def tokenize(self):
        tokenizer = Tokenizer(lower=True, split=" ", num_words= self.max_feature)
        tokenizer.fit_on_texts(self.X_train)
        X_train_vec = tokenizer.texts_to_sequences(self.X_train)
        X_test_vec = tokenizer.texts_to_sequences(self.X_test)
        MAXLEN = np.quantile([len(x) for x in X_train_vec], self.quantile).astype(int)
        return MAXLEN, tokenizer, X_train_vec, X_test_vec
    
    def padding(self):
        MAXLEN, tokenizer, X_train_vec, X_test_vec = self.tokenize()
        X_train_vec = pad_sequences(X_train_vec, maxlen=MAXLEN, padding="post")
        X_test_vec =  pad_sequences(X_test_vec, maxlen=MAXLEN, padding="post")
        return MAXLEN, tokenizer, X_train_vec, X_test_vec
     
    
    def embedding_matrix(self, fb_model):
        _, tokenizer, _, _ = self.tokenize()
        vocab_size = len(tokenizer.word_index) + 1
        num_words = min(self.max_feature, vocab_size)
        embedding_matrix = np.zeros((num_words, 300))
        
        for word, i in tokenizer.word_index.items():
            if i >= self.max_feature:
                continue    

            embedding_vector = fb_model.wv[word]
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
        nonzero_elements = np.count_nonzero(np.count_nonzero(embedding_matrix, axis=1))
        print('proportion of vocabulary coverd by pretrained model:',
              nonzero_elements /self.max_feature)
        return embedding_matrix
    
    def pooling(self, embedding_matrix, dopool, n_split):
        MAXLEN, tokenizer, X_train_vec, X_test_vec = self.padding()
        embedding = Embedding(input_dim= self.max_feature, 
                  output_dim = embedding_matrix.shape[1],
                  weights=[embedding_matrix], trainable=False,
                  input_length= MAXLEN)

        if dopool == 'max':
            pool = GlobalMaxPool1D()
        elif dopool == 'avg':
            pool = GlobalAveragePooling1D()

        X_train_split = np.array_split(X_train_vec, n_split)
        X_test_split = np.array_split(X_test_vec, n_split)
        X_train_emb = []
        X_test_emb = []
        for i in range(len(X_train_split)):
            X_train_emb.append(pool(embedding(X_train_split[i])))
            X_test_emb.append(pool(embedding(X_test_split[i])))
        return np.vstack(X_train_emb), np.vstack(X_test_emb)  
    


#berechnet Gradientbasierte Unsicherheitsmetriken für Deep-Learning
#angewendete Metriken sind L1-, L2-Norm, Max, Min und Mittelwert der Gradienten 
#der negativen Log-Likelihood an der Stelle der prognostizierten Klasse
#Siehe https://arxiv.org/pdf/1805.08440.pdf
class gradient_info(object):
    
    def __init__(self,model):
        self.model = model
        #self.x = np.array([x])
            
    def gradient_calculator(self, x):
        x_vec = np.array([x])
        with tf.GradientTape(persistent=True, watch_accessed_variables=True) as tape:
            y = tf.math.reduce_max(self.model(x_vec))
            m = -tf.math.log(y)
        grads = tape.gradient(m,  self.model.trainable_weights)
        return grads
    
    def gradient_L2Norm(self, grads):
        #grads = self.gradient_calculator()
        return tf.norm(grads[0], ord = 'euclidean').numpy()
    
    def gradient_L1Norm(self, grads):
        #grads = self.gradient_calculator()
        return tf.norm(grads[0], ord = 1).numpy()
    
    def gradient_max(self, grads):
        #grads = self.gradient_calculator()
        return tf.reduce_max(grads[0]).numpy()
                       
    def gradient_min(self, grads):
        #grads = self.gradient_calculator()
        return tf.reduce_min(grads[0]).numpy()
    
    def gradient_mean(self, grads):
        #grads = self.gradient_calculator()
        return tf.reduce_mean(grads[0]).numpy()
    
    def calculate_gradinfo(self, X_matrix):
        L2 = []
        L1 = []
        Max = []
        Min = []
        Mean = []
        for X_ in X_matrix:
            gradient = self.gradient_calculator(X_)
            L2Norm = self.gradient_L2Norm(gradient)
            L1Norm = self.gradient_L1Norm(gradient)
            grad_max = self.gradient_max(gradient)
            grad_min = self.gradient_min(gradient)
            grad_mean = self.gradient_mean(gradient)
            L2.append(L2Norm) 
            L1.append(L1Norm)
            Max.append(grad_max) 
            Min.append(grad_min)
            Mean.append(grad_mean)
        return L2, L1, Max, Min, Mean
                
        
        
#Implementation der Variationalen Scicht      
#Ausführlich siehe: 
#http://krasserm.github.io/2019/03/14/bayesian-neural-networks/
#https://arxiv.org/pdf/1505.05424.pdf
#https://gluon.mxnet.io/chapter18_variational-methods-and-uncertainty/bayes-by-backprop.html
class DenseVariational(Layer):
    def __init__(self,
                 units,
                 kl_weight,
                 activation=None,
                 prior_sigma_1=1.5,
                 prior_sigma_2=0.1,
                 prior_pi=0.5,
                 prior = 'mixture',
                 **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.kl_weight = kl_weight
        self.activation = activation
        self.prior_sigma_1 = prior_sigma_1
        self.prior_sigma_2 = prior_sigma_2
        self.prior_pi_1 = prior_pi
        self.prior_pi_2 = 1.0 - prior_pi
        self.prior = prior
        self.init_sigma = np.sqrt(self.prior_pi_1 * self.prior_sigma_1 ** 2 + \
                                  self.prior_pi_2 * self.prior_sigma_2 ** 2) 
        
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'units': self.units,
            'kl_weight': self.kl_weight,
            'activation': self.activation,
            'prior_sigma_1': self.prior_sigma_1,
            'prior_sigma_2': self.prior_sigma_2,
            'prior_pi': self.prior_pi_1,
            'prior': self.prior
        })
        return config
    

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.units

    def build(self, input_shape):
        sigma = np.sqrt(self.prior_pi_1 * self.prior_sigma_1 ** 2 + \
                                  self.prior_pi_2 * self.prior_sigma_2 ** 2) 
        self.kernel_mu = self.add_weight(name='kernel_mu',
                                         shape=(input_shape[1], self.units),
                                         initializer=initializers.RandomNormal\
                                         (stddev= self.init_sigma, seed = 1234),
                                         trainable=True)
        self.bias_mu = self.add_weight(name='bias_mu',
                                       shape=(self.units,),
                                       initializer=initializers.RandomNormal\
                                       (stddev=self.init_sigma, seed = 1234),
                                       trainable=True)
        self.kernel_rho = self.add_weight(name='kernel_rho',
                                          shape=(input_shape[1], self.units),
                                          initializer=initializers.constant(0.0),
                                          trainable=True)
        self.bias_rho = self.add_weight(name='bias_rho',
                                        shape=(self.units,),
                                        initializer=initializers.constant(0.0),
                                        trainable=True)
        super().build(input_shape)

    def call(self, inputs, **kwargs):
        kernel_sigma = tf.math.softplus(self.kernel_rho)
        kernel = self.kernel_mu + kernel_sigma * tf.random.normal(self.kernel_mu.shape)

        bias_sigma = tf.math.softplus(self.bias_rho)
        bias = self.bias_mu + bias_sigma * tf.random.normal(self.bias_mu.shape)

        self.add_loss(self.kl_loss(kernel, self.kernel_mu, kernel_sigma) +
                      self.kl_loss(bias, self.bias_mu, bias_sigma))
        
        activation = activations.get(self.activation)
        return activation(K.dot(inputs, kernel) + bias)

    def kl_loss(self, w, mu, sigma):
        variational_dist = tfp.distributions.Normal(mu, sigma)
        return self.kl_weight * K.sum(variational_dist.log_prob(w) - self.log_prior_prob(w))

    def log_prior_prob(self, w):
        
        if self.prior == 'mixture':
            comp_1_dist = tfp.distributions.Normal(0.0, self.prior_sigma_1)
            comp_2_dist = tfp.distributions.Normal(0.0, self.prior_sigma_2)
            dist =  K.log(self.prior_pi_1 * comp_1_dist.prob(w) +
                         self.prior_pi_2 * comp_2_dist.prob(w))
            
        elif self.prior == 'gaussian':
            comp_dist = tfp.distributions.Normal(0.0, self.prior_sigma_1)
            dist = K.log(comp_dist.prob(w))
        return dist

               
#Erstellt RNN-Netzwerke
#Erstellt RNN-Netzwerke
def RNN_custom(param_embed, param_conv, 
                param_lstm, prior_params, 
                param_optimizer, param_fit, 
                method, run_vi,
                conv_layer, embedding_matrix, 
                training, MAXLEN,
                run_dropout2, training2,
                 train_size):
     
    inputs = Input(shape=(MAXLEN, ))
    x = Embedding(input_dim = param_embed['input_dim'], 
                  output_dim = param_embed['output_dim'], 
                  weights= [embedding_matrix],
                  trainable = param_embed['trainable'])(inputs)

    if conv_layer:
        x = Conv1D(filters = param_conv['filters'], 
                   kernel_size = param_conv['kernel_size'], 
                   padding= param_conv['padding'], 
                   activation = param_conv['activation_conv'], 
                   strides= param_conv['strides'],
                   use_bias= param_conv['use_bias'],
                   kernel_initializer= param_conv['kernel_initializer'], 
                   bias_initializer= param_conv['bias_initializer'])(x)
        x = MaxPooling1D(pool_size = param_conv['pool_size'])(x)
        x = SpatialDropout1D(rate = param_conv\
                  ['rate_spatialDropout'])(x, training = training)
    
    if method == 'lstm':
        x = LSTM(units = param_lstm['units'],
                 activation= param_lstm['activation'],
                 recurrent_activation= param_lstm['recurrent_activation'],
                 use_bias= param_lstm['use_bias'],
                 kernel_initializer =param_lstm['kernel_initializer'],
                 recurrent_initializer= param_lstm['recurrent_initializer'],
                 bias_initializer=param_lstm['bias_initializer'],
                 dropout= param_lstm['dropout'],
                 recurrent_dropout= param_lstm['recurrent_dropout'],
                 return_sequences= False
                )(x, training = training)
        

    elif method == 'bi-lstm':
        x = Bidirectional(LSTM(units = param_lstm['units'],
                 activation= param_lstm['activation'],
                 recurrent_activation= param_lstm['recurrent_activation'],
                 use_bias= param_lstm['use_bias'],
                 kernel_initializer =param_lstm['kernel_initializer'],
                 recurrent_initializer= param_lstm['recurrent_initializer'],
                 bias_initializer=param_lstm['bias_initializer'],
                 dropout= param_lstm['dropout'],
                 recurrent_dropout= param_lstm['recurrent_dropout'],
                 return_sequences= False
                ))(x, training = training)
         
    elif method == 'gru':
        x = GRU(units = param_lstm['units'],
                 activation= param_lstm['activation'],
                 recurrent_activation= param_lstm['recurrent_activation'],
                 use_bias= param_lstm['use_bias'],
                 kernel_initializer =param_lstm['kernel_initializer'],
                 recurrent_initializer= param_lstm['recurrent_initializer'],
                 bias_initializer=param_lstm['bias_initializer'],
                 dropout= param_lstm['dropout'],
                 recurrent_dropout= param_lstm['recurrent_dropout'],
                 return_sequences= False
                )(x, training = training)
          
    elif method == 'bi-gru':
        x = Bidirectional(GRU(units = param_lstm['units'],
                 activation= param_lstm['activation'],
                 recurrent_activation= param_lstm['recurrent_activation'],
                 use_bias= param_lstm['use_bias'],
                 kernel_initializer =param_lstm['kernel_initializer'],
                 recurrent_initializer= param_lstm['recurrent_initializer'],
                 bias_initializer=param_lstm['bias_initializer'],
                 dropout= param_lstm['dropout'],
                 recurrent_dropout= param_lstm['recurrent_dropout'],
                 return_sequences= False
                ))(x, training = training) 
    
    if run_dropout2:
        x = Dropout(0.5)(x, training=training2)
        
    if run_vi:
        #batch_size = param_fit['batch_size']
        #num_batches = train_size / batch_size
        kl_weight = 1.0 / train_size
        outputs = DenseVariational(units = 100, 
                                 kl_weight = kl_weight, 
                                 **prior_params, 
                                activation='softmax')(x)
        
            
    else:
        outputs = Dense(units = 100, 
                        activation='softmax',
                        use_bias = param_lstm['use_bias'],
                        kernel_initializer = param_lstm['kernel_initializer'],
                        bias_initializer = param_lstm['bias_initializer'])(x)
        
    model = Model(inputs, outputs)
    adam = tf.keras.optimizers.Adam(learning_rate= param_optimizer['learning_rate'], 
                                    beta_1= param_optimizer['beta_1'], 
                                    beta_2= param_optimizer['beta_2'], 
                                    epsilon= param_optimizer['epsilon'], 
                                    name= param_optimizer['name'])

    model.compile(loss= 'categorical_crossentropy',
                  optimizer= adam,
                  metrics=['accuracy'])
    return model

#Erstellt CNN-Netzwerke
def CNN_custom(param_embed, param_conv, prior_params, 
                param_optimizer, param_fit, run_vi,
                embedding_matrix, training, MAXLEN,
                train_size):
    
    inputs = Input(shape=(MAXLEN, ))
    x = Embedding(input_dim = param_embed['input_dim'], 
                  output_dim = param_embed['output_dim'], 
                  weights= [embedding_matrix],
                  trainable = param_embed['trainable'])(inputs)
        
    x = Conv1D(filters = param_conv['filters'], 
                   kernel_size = param_conv['kernel_size'], 
                   padding= param_conv['padding'], 
                   activation = param_conv['activation_conv'], 
                   strides= param_conv['strides'],
                   use_bias= param_conv['use_bias'],
                   kernel_initializer= param_conv['kernel_initializer'], 
                   bias_initializer= param_conv['bias_initializer'])(x)
    
    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPool1D()(x)
    x = concatenate([avg_pool, max_pool])
    x = Dropout(rate = param_conv['dropout'])(x, training = training)
    x = Dense(units = 100, 
              activation= param_conv['activation_conv'],
              use_bias = param_conv['use_bias'],
              kernel_initializer = param_conv['kernel_initializer'],
              bias_initializer = param_conv['kernel_initializer'])(x)
    
    if run_vi:
        #batch_size = param_fit['batch_size']
        #num_batches = train_size / batch_size
        kl_weight = 1.0 / train_size
        outputs = DenseVariational(units = 100, 
                                 kl_weight = kl_weight, 
                                 **prior_params, 
                                activation='softmax')(x)
            
    else:
        outputs = Dense(units = 100, 
                        activation='softmax',
                        use_bias = param_conv['use_bias'],
                        kernel_initializer = param_conv['kernel_initializer'],
                        bias_initializer = param_conv['bias_initializer'])(x)
        
    model = Model(inputs, outputs)
    adam = tf.keras.optimizers.Adam(learning_rate= param_optimizer['learning_rate'], 
                                    beta_1= param_optimizer['beta_1'], 
                                    beta_2= param_optimizer['beta_2'], 
                                    epsilon= param_optimizer['epsilon'], 
                                    name= param_optimizer['name'])

    model.compile(loss= 'categorical_crossentropy',
                  optimizer= adam,
                  metrics=['accuracy'])
    return model


#Führt Monte-Carlo-Simulation durch
def run_mc(model,sample_num, X_test,  y_test):
    yhat = np.zeros((y_test.shape[0],sample_num))
    y_test_ = y_test.argmax(axis = 1)
    sample = np.zeros((sample_num,y_test.shape[0],y_test.shape[1]))

    for j in range(sample_num):
        sample[j,:,:] = model.predict(X_test)
        yhat[:,j] = sample[j,:,:].argmax(axis = 1).astype(int)
    #pred_prob = np.apply_along_axis\
    #(lambda x: np.unique(x, return_counts=True)[1], axis=1, arr= yhat)/yhat.shape[1]
    hard_pred = np.zeros((yhat.shape[0], 100)).astype(int)
    for i in range(yhat.shape[0]):
        x = list(yhat[i])
        d = []
        for j in range(100):
            d.append( x.count(j))
        hard_pred[i] = d
    pred_prob_mode = hard_pred/yhat.shape[1] 
    #yhat_mc = pred_prob.argmax(axis = 1).astype(int)
    #acc = accuracy_score(y_test_, yhat_mc)
    pred_prob_mean =  sample.mean(axis = 0)
    return pred_prob_mode, pred_prob_mean, sample


def prepare_metaclf(y_pred, metrics, y_test):
    test_label = y_test.argmax(axis = 1)
    yhat = np.argmax(y_pred, axis=1)
    metrics_r = metrics[yhat == test_label]
    metrics_w = metrics[yhat != test_label]
    safe, risky = metrics_r, metrics_w
    labels = np.zeros((safe.shape[0] + risky.shape[0]), dtype=np.int32)
    labels[:safe.shape[0]] += 1
    examples = np.concatenate((safe, risky), axis=None)
    #auroc = sk.roc_auc_score(labels, examples)
    return examples, labels


#Erstellt die Ergebnisse aus Metaklassifikationsmodell der logistischen Regression
def metaclassif(features, labels):
    feature_train, feature_test, label_train, label_test = \
    model_selection.train_test_split(features, 
                                     labels, 
                                     test_size =  0.2, 
                                     random_state = 123,
                                    shuffle = True)
    Model = LogisticRegression()
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    space = dict()
    space['solver'] = ['lbfgs', 'liblinear', 'sag']
    space['penalty'] = ['none', 'l1', 'l2']
    space['C'] = [1e-3, 1e-2, 1e-1, 1, 10, 100]
    search = GridSearchCV(Model, space, 
                          scoring='roc_auc', 
                          n_jobs=-1,verbose = 1, cv=cv)

    lr = search.fit(feature_train, label_train)
    auroc = roc_auc_score(label_test, 
                          lr.predict_proba(feature_test)[:,1])
    return  auroc, label_test, feature_test, lr


#Berechnet die finale Ergebnisse der Metaklassifikation
def result_final(pred_prob, y_test):
    test_label = y_test.argmax(axis = 1)
    pred_class_prob = pred_prob.max(axis = 1)
    entropy = (1/np.log(100))*np.sum(pred_prob*\
            np.log(np.abs(pred_prob) + 1e-11), axis = 1)

    prob, labels = prepare_metaclf\
    (y_pred = pred_prob, metrics = pred_class_prob, y_test = y_test)

    entropy_ , _ = prepare_metaclf\
    (y_pred = pred_prob, metrics = entropy, y_test = y_test)

    features = np.array([prob, entropy_]).T
    auroc_metaclf, label_test_meta, \
    feature_test_meta, lr = metaclassif(features, labels)

    Accuracy = [accuracy_score(pred_prob.argmax(axis = 1), test_label)]
    auroc_final = [sk.roc_auc_score(label_test_meta, \
                feature_test_meta[:,i]) for i in range(feature_test_meta.shape[1])]
    return Accuracy, auroc_final, auroc_metaclf, \
           label_test_meta, feature_test_meta, lr


#Bereschnet den kritischen Wert des Kolmogorov-Smrinoff-Tests
def criticalvalue_KS(alpha, n,m):
    return np.sqrt(-np.log(alpha*0.5)*0.5)*np.sqrt((m+n)/(m*n))
    
