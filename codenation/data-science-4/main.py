#!/usr/bin/env python
# coding: utf-8

# # Desafio 6
# 
# Neste desafio, vamos praticar _feature engineering_, um dos processos mais importantes e trabalhosos de ML. Utilizaremos o _data set_ [Countries of the world](https://www.kaggle.com/fernandol/countries-of-the-world), que contém dados sobre os 227 países do mundo com informações sobre tamanho da população, área, imigração e setores de produção.
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Setup_ geral

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import sklearn as sk
from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import (
    CountVectorizer, TfidfTransformer, TfidfVectorizer
)
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder, Binarizer, KBinsDiscretizer,
    MinMaxScaler, StandardScaler, PolynomialFeatures
)


# In[2]:


# Algumas configurações para o matplotlib.
#%matplotlib inline

#from IPython.core.pylabtools import figsize


#figsize(12, 8)

#sns.set()


# In[3]:


countries = pd.read_csv("countries.csv")


# In[4]:


new_column_names = [
    "Country", "Region", "Population", "Area", "Pop_density", "Coastline_ratio",
    "Net_migration", "Infant_mortality", "GDP", "Literacy", "Phones_per_1000",
    "Arable", "Crops", "Other", "Climate", "Birthrate", "Deathrate", "Agriculture",
    "Industry", "Service"
]

countries.columns = new_column_names

countries.head(5)


# ## Observações
# 
# Esse _data set_ ainda precisa de alguns ajustes iniciais. Primeiro, note que as variáveis numéricas estão usando vírgula como separador decimal e estão codificadas como strings. Corrija isso antes de continuar: transforme essas variáveis em numéricas adequadamente.
# 
# Além disso, as variáveis `Country` e `Region` possuem espaços a mais no começo e no final da string. Você pode utilizar o método `str.strip()` para remover esses espaços.

# ## Inicia sua análise a partir daqui

# In[5]:


# Sua análise começa aqui.
countries.info()


# In[6]:


#Alterando o separador decimal do data frame para ponto:
for i in new_column_names:
    if countries[i].dtype != 'object':
        continue
    if i not in ['Country', 'Region']:
        countries[i] = countries[i].str.replace(',', '.').astype('float64')


# In[7]:


#Verificando a alteração:
countries.head(5)


# In[8]:


#Removendo os espaços das variáveis Country e Region.
countries['Country'] = countries.Country.apply(lambda i: i.strip())
countries['Region'] = countries.Region.apply(lambda i: i.strip())


# In[9]:


countries['Country'][0]


# In[10]:


countries['Region'][0]


# ## Questão 1
# 
# Quais são as regiões (variável `Region`) presentes no _data set_? Retorne uma lista com as regiões únicas do _data set_ com os espaços à frente e atrás da string removidos (mas mantenha pontuação: ponto, hífen etc) e ordenadas em ordem alfabética.

# In[11]:


def q1():
    # Retorne aqui o resultado da questão 1.
    regions = sorted(countries.Region.unique())
    return regions
q1()


# ## Questão 2
# 
# Discretizando a variável `Pop_density` em 10 intervalos com `KBinsDiscretizer`, seguindo o encode `ordinal` e estratégia `quantile`, quantos países se encontram acima do 90º percentil? Responda como um único escalar inteiro.

# In[12]:


def q2():
    # Retorne aqui o resultado da questão 2.
    discretizer = KBinsDiscretizer(n_bins = 10, encode = 'ordinal', strategy = 'quantile')
    discretizer_pop = discretizer.fit_transform(countries[['Pop_density']])
    above_p90 = discretizer.bin_edges_[0][9]
    return int(countries[countries['Pop_density'] >= above_p90]['Pop_density'].count())
q2()


# # Questão 3
# 
# Se codificarmos as variáveis `Region` e `Climate` usando _one-hot encoding_, quantos novos atributos seriam criados? Responda como um único escalar.

# In[13]:


def q3():
    # Retorne aqui o resultado da questão 3.
    regions_ntotal = countries.Region.unique().size
    climate_ntotal = countries.Climate.unique().size
    return regions_ntotal + climate_ntotal
q3()


# ## Questão 4
# 
# Aplique o seguinte _pipeline_:
# 
# 1. Preencha as variáveis do tipo `int64` e `float64` com suas respectivas medianas.
# 2. Padronize essas variáveis.
# 
# Após aplicado o _pipeline_ descrito acima aos dados (somente nas variáveis dos tipos especificados), aplique o mesmo _pipeline_ (ou `ColumnTransformer`) ao dado abaixo. Qual o valor da variável `Arable` após o _pipeline_? Responda como um único float arredondado para três casas decimais.

# In[14]:


test_country = [
    'Test Country', 'NEAR EAST', -0.19032480757326514,
    -0.3232636124824411, -0.04421734470810142, -0.27528113360605316,
    0.13255850810281325, -0.8054845935643491, 1.0119784924248225,
    0.6189182532646624, 1.0074863283776458, 0.20239896852403538,
    -0.043678728558593366, -0.13929748680369286, 1.3163604645710438,
    -0.3699637766938669, -0.6149300604558857, -0.854369594993175,
    0.263445277972641, 0.5712416961268142
]


# In[15]:


#criando uma cópia do data frame
countreis_c = countries.copy()


# In[16]:


def q4():
    # Retorne aqui o resultado da questão 4.
    #criando a instância para o pipeline
    pipeline_countries = Pipeline(steps = [
    ("imputer", SimpleImputer(strategy = "median")),
    ("standard", StandardScaler())])
    pipeline_countries.fit(countreis_c.iloc[:,2:])
    transform_test_counrty = pipeline_countries.transform([test_country[2:]])
    return float(round(transform_test_counrty[0, countreis_c.columns.get_loc('Arable') - 2], 3))
q4()    


# ## Questão 5
# 
# Descubra o número de _outliers_ da variável `Net_migration` segundo o método do _boxplot_, ou seja, usando a lógica:
# 
# $$x \notin [Q1 - 1.5 \times \text{IQR}, Q3 + 1.5 \times \text{IQR}] \Rightarrow x \text{ é outlier}$$
# 
# que se encontram no grupo inferior e no grupo superior.
# 
# Você deveria remover da análise as observações consideradas _outliers_ segundo esse método? Responda como uma tupla de três elementos `(outliers_abaixo, outliers_acima, removeria?)` ((int, int, bool)).

# In[17]:


def q5():
    # Retorne aqui o resultado da questão 4.
    migration = np.array(countries['Net_migration'].dropna())
    q_one = np.quantile(migration, 0.25)
    q_three = np.quantile(migration, 0.75)
    high_inter = q_three + 1.5*(q_three - q_one)
    low_inter = q_one - 1.5*(q_three - q_one)
    high_outliers = (migration > high_inter).sum()
    low_outliers = (migration < low_inter).sum()
    return (int(low_outliers), int(high_outliers), False)
q5()


# Os valores da variável Net-migration representam a diferença entre a entrada e saída de pessoas de um país. Essa taxa possui valores dentro do conjunto dos números inteiros (Z), ou seja, poderá ter valores positivos e negativos. Um valor positivo representa que mais pessoas entram do que saem de um país. Um valor negativo significa que saem mais pessoas do que entram em um país. Quando tem o valor zero, significa que há um equilíbrio. Logo, dentro do contexto dos dados, os outliers identificados representam paises que possuem uma grande entrada de pessoas em relação a outros paises. São diferentes fatores que podem fazer essa taxa variar. A identificação de erros teria que ser avaliada com base em vários outros parâmetros. Por isso não removeria esses dados do modelo.

# ## Questão 6
# Para as questões 6 e 7 utilize a biblioteca `fetch_20newsgroups` de datasets de test do `sklearn`
# 
# Considere carregar as seguintes categorias e o dataset `newsgroups`:
# 
# ```
# categories = ['sci.electronics', 'comp.graphics', 'rec.motorcycles']
# newsgroup = fetch_20newsgroups(subset="train", categories=categories, shuffle=True, random_state=42)
# ```
# 
# 
# Aplique `CountVectorizer` ao _data set_ `newsgroups` e descubra o número de vezes que a palavra _phone_ aparece no corpus. Responda como um único escalar.

# In[18]:


#Carregando o dataset newgroups:

categories = ['sci.electronics', 'comp.graphics', 'rec.motorcycles']
newsgroup = fetch_20newsgroups(subset="train", categories=categories, shuffle=True, random_state=42)
#newsgroup.data


# In[19]:


def q6():
    # Retorne aqui o resultado da questão 4.
    #Criando a instância:
    vectorizer = CountVectorizer()
    #Apndendo o vocabulário da variável categories e retornando uma matriz document-term:
    matrix_newsgruop = vectorizer.fit_transform(newsgroup.data)
    counts = pd.DataFrame(matrix_newsgruop.toarray(),
                          columns=vectorizer.get_feature_names())
    return int(counts['phone'].sum())
q6()


# ## Questão 7
# 
# Aplique `TfidfVectorizer` ao _data set_ `newsgroups` e descubra o TF-IDF da palavra _phone_. Responda como um único escalar arredondado para três casas decimais.

# In[20]:


def q7():
    # Retorne aqui o resultado da questão 4.
    #Criando a instância:
    vectorizer = TfidfVectorizer()
    #Apndendo o vocabulário da variável categories e o IDF, retornando uma matriz document-term:
    matrix_idf = vectorizer.fit_transform(newsgroup.data)
    counts_idf = pd.DataFrame(matrix_idf.toarray(),
                              columns=vectorizer.get_feature_names())
    return float(round(counts_idf['phone'].sum(),3))
q7()


# In[ ]:




