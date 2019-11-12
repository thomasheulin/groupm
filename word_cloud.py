#------------------------------------------------------------------------------------------------------------------------------

Nuage de mots

Porgramme permettant, à partir d'une étude sur des avis de publicités accompagnés d'une note (4 modalités), d'extraire les mots 
ou groupes de mots les plus les mots les plus pertinents en positif ou en négatif et synthétiser le tout dans un fichier .csv.
L'étude est réalisée sur plusieurs médias différent (TV, Presse, Web)
La méthode utilisée est la régression logistique.

Le fichier produit par le programme sert ensuite de base à un autre programme qui permet de visualser le nuage de mots de façon
interactive, en choissant le type de média, mots positifs ou négatifs, et retrouver un mot dans ses contextes.

#------------------------------------------------------------------------------------------------------------------------------


import pandas as pd
from nltk.corpus import stopwords
import re
import unidecode
from cltk.lemmatize.french.lemma import LemmaReplacer
from french_lefff_lemmatizer.french_lefff_lemmatizer import FrenchLefffLemmatizer
import json

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split

#------------------------------------------------------------------------------------------------------------------------------

PATH_IMPUT_FILE = 'data.xlsx'
PATH_OUTPUT = 'wordcloud'
PATH_SETUP_FILE = 'config\\setup.json'


#------------------------------------------------------------------------------------------------------------------------------


french_stop_words = list(set(stopwords.words('french')) - set(['pas', 'non']))

lemmatizer = FrenchLefffLemmatizer(load_only_pos=['a', 'n'])

# chargement du fichier de config
# string -> dictionnaire
def load_setup(path_setup_file):

    with open(path_setup_file) as json_file:
        data = json.load(json_file)
        
    return data

# nettoyage des lignes du df selon les paramètres de config
def setup_clean_line(line, setup_data):
    x = line.split()
    
    for i in range(len(x)):
        for s in setup_data['clean_data_pos']:
            if x[i] in setup_data['clean_data_pos'][s]:
                x[i] = s
                
    for i in range(len(x)):
        for s in setup_data['clean_data_neg']:
            if x[i] in setup_data['clean_data_neg'][s]:
                x[i] = s
        
    return ' '.join(x)

# nettoyage des lignes du df selon les paramètres de config
def setup_clean_list(f, setup_data):
    
    f = [setup_clean_line(line, setup_data) for line in f]
    
    return f

# suppression des mots indésirables du df selon les paramètres de config
def filter_df(dataframe, setup_data):
    data_result = []
    
    for i in range(len(dataframe['word'])):
        row = dataframe.iloc[i, :].tolist()
        
        if row[1] > 0:
            if row[0] in setup_data['replace_results_pos']:
                data_result.append([setup_data['replace_results_pos'][row[0]], row[1]])
                
            elif row[0] not in setup_data['stop_results_pos']:
                data_result.append([row[0], row[1]])
                
        if row[1] < 0:
            if row[0] in setup_data['replace_results_neg']:
                data_result.append([setup_data['replace_results_neg'][row[0]], row[1]])
                
            elif row[0] not in setup_data['stop_results_neg']:
                data_result.append([row[0], row[1]])
                
    c = ['word', 'coefficient']

    df_result = pd.DataFrame([], columns = c)

    df_result = df_result.append(pd.DataFrame(data_result,  columns = c))
    
    return df_result

# suppression des mots de taille < 2
def erase_chars(s):
    r = []

    for x in s.split():
        if len(x) > 2:
            r.append(x)
    
    return ' '.join(r)

# nettoyage de la ponctuation
def preprocess(line):
    chars = ['.', ',', ';', ':', '!', '?', '\'', '(', ')', '[', ']']
    
    line = str(line).lower()
    line = unidecode.unidecode(line)
    
    for c in chars:
        line = line.replace(c, ' ')
        
    line = erase_chars(line)
        
    return ' '.join(line.split())

# nettoyage de la ponctuation
def preprocess_features(f):
    
    f = [preprocess(line) for line in f]
    
    return f

# nettoyage des stop words
def remove_stop_words(corpus):

    removed_stop_words = []

    for review in corpus:

        removed_stop_words.append(

            ' '.join([word for word in review.split() 

                      if word not in french_stop_words])

        )

    return removed_stop_words

# fonction de lemmatisation
def get_lemmatized_text(corpus):

    return [' '.join([lemmatizer.lemmatize(word) for word in review.split()]) for review in corpus]

# encodage des avis
def targetmap(s):
    if s == 'Pas du tout':
        return 0
    elif s == 'Peu':
        return 0
    elif s == 'Assez':
        return 1
    else: return 1
 
# fonction de recherche de sub string servant à éviter les redondances
def check_substr(string, sub_str): 
    if (string.find(sub_str) == -1): 
        return 0
    else: 
        return 1
        
# fonction de nettoyage des redondances
# df -> df
def clean_duplicates(df):

    c = ['word', 'coefficient']
    df_clean = pd.DataFrame([], columns = c)

    for s in [1, -1]:
    
        s_df = df.loc[s*df['coefficient'] > 0, :]

        clean_index = []

        for i in range(len(s_df)):
            for j in range(len(s_df)):
                if i != j and check_substr(s_df.iloc[i, 0], s_df.iloc[j, 0]) == 1:
                    clean_index.append(j)
        
        s_df = s_df.drop(s_df.index[clean_index])
        
        df_clean = df_clean.append(s_df, ignore_index = True)
        
    return df_clean
    
WORDS_LIMIT = load_setup(PATH_SETUP_FILE)['word_limit']

df = pd.read_excel(PATH_IMPUT_FILE)

df = df.rename(columns = {'Media': 'media', 'Agrement': 'agrement', 'Raisons': 'raisons'})
df = df.loc[:,['media', 'agrement', 'raisons']]

media_types = list(set(df.loc[:, 'media'].tolist()))

training_matrix = []

# prepare la donnée pour chaque média, stocke les dataframes dans training_matrix
for media_type in media_types:
    
    feature_list = df.loc[df['media'].isin([media_type]),'raisons'].tolist()
    label_list = df.loc[df['media'].isin([media_type]),'agrement'].tolist()
    
    label_list = [targetmap(item) for item in label_list]
    
    feature_list_clean = preprocess_features(feature_list)
    no_stop_words = remove_stop_words(feature_list_clean)
    lemmatized_list = get_lemmatized_text(no_stop_words)
    lemmatized_list = setup_clean_list(lemmatized_list, load_setup(PATH_SETUP_FILE))
    
    df_feature_data = {'feature': feature_list, 'agrement': label_list} 
    df_feature = pd.DataFrame(df_feature_data)
    
    training_matrix.append([media_type, lemmatized_list, label_list, df_feature])

clean_matrix = []    

# créé le modèle de régression pour chaque média
for training in training_matrix:
    
    print('-->>', training[0])
    
    target = training[2]
    lemmatized_list = training[1]
    
    ngram_vectorizer = CountVectorizer(binary=True, ngram_range=(1, 2))

    ngram_vectorizer.fit(lemmatized_list)

    X = ngram_vectorizer.transform(lemmatized_list)

    
    final_ngram = LogisticRegression(C=1)

    final_ngram.fit(X, target)

    feature_to_coef = {word: coef for word, coef in zip(ngram_vectorizer.get_feature_names(), final_ngram.coef_[0])}
    
    # df contenant les mots trouvés et leurs coéfficient
    df_final_data = []
    
    # mots positifs
    for d in sorted(feature_to_coef.items(), key=lambda x: x[1], reverse=True)[:WORDS_LIMIT]:
        df_final_data.append([d[0], d[1]])
    
    # mots négatifs
    for d in sorted(feature_to_coef.items(), key=lambda x: x[1])[:WORDS_LIMIT]:
        df_final_data.append([d[0], d[1]])

    c = ['word', 'coefficient']

    df_final = pd.DataFrame([], columns = c)

    df_final = df_final.append(pd.DataFrame(df_final_data,  columns = c))
    
    df_final = filter_df(df_final, load_setup(PATH_SETUP_FILE))
    
    df_final.to_csv(PATH_OUTPUT + '\\' + training[0] + '.csv')
    
    # etape de nettoyage selon la config
    
    df_clean = clean_duplicates(df_final)
    
    df_media_data = {'raisons': training[1], 'agrement': training[2]} 
    df_media = pd.DataFrame(df_media_data)
    
    clean_matrix.append([training[0], df_clean, df_media, training[3]])
    
    print(df_final)
    
# etape finale : compter le nombre d'occurence de chaque mot fournit par le modele dans la donnee d'origine

df_wordcloud_data = []

for matrix in clean_matrix:
    clean_models = matrix[1]
    clean_raisons = matrix[2]
    clean_features = matrix[3]
    
    for s in [1, 0]:
        
        if s == 1:
            models = clean_models.loc[clean_models['coefficient'] > 0, 'word'].tolist()
        else:
            models = clean_models.loc[clean_models['coefficient'] < 0, 'word'].tolist()

        raisons = clean_raisons.loc[clean_raisons['agrement'] == s, "raisons"].tolist()
        feature = clean_features.loc[clean_features['agrement'] == s, "feature"].tolist()
    

        for e in models:
            ws = sum(e in s for s in raisons)
            for f in raisons:
                if e in f:
                    df_wordcloud_data.append([matrix[0], s, e, ws, feature[raisons.index(f)]])


c_wordcloud = ['media', 'positive', 'word', 'word_count', 'raison']

df_wordcloud = pd.DataFrame(df_wordcloud_data, columns = c_wordcloud)   
df_wordcloud.to_csv(PATH_OUTPUT + '\\wordcloud.csv')
