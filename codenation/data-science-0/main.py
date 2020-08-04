#!/usr/bin/env python
# coding: utf-8

# # Desafio 1
# 
# Para esse desafio, vamos trabalhar com o data set [Black Friday](https://www.kaggle.com/mehdidag/black-friday), que reúne dados sobre transações de compras em uma loja de varejo.
# 
# Vamos utilizá-lo para praticar a exploração de data sets utilizando pandas. Você pode fazer toda análise neste mesmo notebook, mas as resposta devem estar nos locais indicados.
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Set up_ da análise

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


black_friday = pd.read_csv("black_friday.csv")


# ## Inicie sua análise a partir daqui

# In[3]:


#Visualizando as 10 primeiras linhas do Data Frame:

black_friday.head(10)


# In[4]:


#Analisando informações do Data Frame Black Friday

black_friday.info()


# In[5]:


#Calculando o percentual de valores faltantes:

x = 537577*0.01
y = 537577 - 379591
w = 537577 - 164278
P2 = int(y/x)
P3 = int(w/x)
print('- O percentual de valores nulos da variável Product_Category_2 é',P2,'%.')
print('- O percentual de valores nulos da variável Product_Category_3 é',P3,'%.')


# Podemos observar nas informações acima que o data frame Black Friday:
# - O data frame possui 537577 linhas e 12 colunas;
# - Os tipo de variáveis presentes são inteiro (5), objeto (5) e float (2);
# - As colunas Product_Category_2 e Product_Category_3 apresentam, respectivamente, 29% e 69% de valores faltantes;
# - A coluna Age (idade) tem valores expressos por intervalos de idades;
# - A coluna Stay_In_Current_City_Years (tempo atual de permanencia na cidade) é representada por valores numéricos do quais, alguns deles, apresentam o símbolo '+' para indicar um tempo maior que o número expresso. Por exemplo: '4+'.
# 

# In[6]:


#Observando dados descritivos de variáveis numéricas

black_friday.describe()


# In[7]:


#Analisando a frequencia dos dados em cada variável

black_friday['Occupation'].value_counts()


# In[8]:


black_friday['Marital_Status'].value_counts()


# In[9]:


black_friday['Gender'].value_counts()


# In[10]:


black_friday['Age'].value_counts()


# In[11]:


black_friday['City_Category'].value_counts()


# In[12]:


black_friday['Stay_In_Current_City_Years'].value_counts()


# In[13]:


black_friday['Product_Category_1'].value_counts()


# In[14]:


black_friday['Product_Category_2'].value_counts()


# In[15]:


black_friday['Product_Category_3'].value_counts()


# In[16]:


black_friday['Purchase'].value_counts()


# In[17]:


#Analisando as colunas Age e Purchase

black_friday[['Age','Purchase']].groupby('Age').sum().sort_values(by='Purchase', ascending=False)


# In[18]:


#Analisando as colunas Gender e Purchase

black_friday[['Gender','Purchase']].groupby('Gender').sum().sort_values(by='Purchase', ascending=False)


# In[19]:


#Analisando as colunas Marital_Status e Purchase

black_friday[['Marital_Status','Purchase']].groupby('Marital_Status').sum().sort_values(by='Purchase', ascending=False)


# In[20]:


#Analisando as colunas Occupation e Purchase

black_friday[['Occupation','Purchase']].groupby('Occupation').sum().sort_values(by='Purchase', ascending=False)


# In[21]:


#Analisando as colunas Stay_In_Current_City_Years e Purchase

black_friday[['Stay_In_Current_City_Years','Purchase']].groupby('Stay_In_Current_City_Years').sum().sort_values(by='Purchase', ascending=False)


# In[22]:


#Analisando as colunas Product_Category_1 e Purchase

black_friday[['Product_Category_1','Purchase']].groupby('Product_Category_1').sum().sort_values(by='Purchase', ascending=False)


# In[23]:


#Analisando as colunas Product_Category_2 e Purchase

black_friday[['Product_Category_2','Purchase']].groupby('Product_Category_2').sum().sort_values(by='Purchase', ascending=False)


# In[24]:


#Analisando as colunas Product_Category_3 e Purchase

black_friday[['Product_Category_3','Purchase']].groupby('Product_Category_3').sum().sort_values(by='Purchase', ascending=False)


# In[25]:


#Analisando as colunas City_Category e Purchase

black_friday[['City_Category','Purchase']].groupby('City_Category').sum().sort_values(by='Purchase', ascending=False)


# ### Normalizando os dados da variável Purchase

# In[26]:


#Criando o data frame com os dados normalizados:

Purchase_Normalized = pd.DataFrame({'Purchase_Nor': ((black_friday['Purchase'] - black_friday.Purchase.min())/(black_friday.Purchase.max() - black_friday.Purchase.min()))})


# In[27]:


Purchase_Normalized.head(10)


# In[28]:


Purchase_Normalized.mean()


# ### Padronizando os dados da variável Purchase

# In[29]:


Purchase_Standardization = pd.DataFrame({'Purchase_STD': ((black_friday['Purchase'] - black_friday.Purchase.mean())/(black_friday.Purchase.std()))})


# In[30]:


Purchase_Standardization.head(10)


# In[31]:


Purchase_Standardization['Purchase_STD'].between(-1,1).sum()


# In[32]:


black_friday.query("Product_Category_2 == 'NaN' & Product_Category_3 == 'NaN'")


# In[33]:


black_friday.shape[0]


# In[34]:


black_friday['Product_Category_3'].dropna().shape[0]


# ## Questão 1
# 
# Quantas observações e quantas colunas há no dataset? Responda no formato de uma tuple `(n_observacoes, n_colunas)`.

# In[35]:


def q1():
    # Retorne aqui o resultado da questão 1.
    return black_friday.shape
q1()


# ## Questão 2
# 
# Há quantas mulheres com idade entre 26 e 35 anos no dataset? Responda como um único escalar.

# In[36]:


def q2():
    # Retorne aqui o resultado da questão 2.
    return black_friday.query('Gender == "F" & Age == "26-35"').shape[0]
q2()


# ## Questão 3
# 
# Quantos usuários únicos há no dataset? Responda como um único escalar.

# In[37]:


def q3():
    # Retorne aqui o resultado da questão 3.
    return black_friday.User_ID.nunique()
q3()


# ## Questão 4
# 
# Quantos tipos de dados diferentes existem no dataset? Responda como um único escalar.

# In[38]:


def q4():
    # Retorne aqui o resultado da questão 4.
    return black_friday.dtypes.nunique()
q4()


# ## Questão 5
# 
# Qual porcentagem dos registros possui ao menos um valor null (`None`, `ǸaN` etc)? Responda como um único escalar entre 0 e 1.

# In[39]:


def q5():
    # Retorne aqui o resultado da questão 5.
    return ((black_friday.shape[0] - black_friday.dropna().shape[0]) / black_friday.shape[0])
q5()


# ## Questão 6
# 
# Quantos valores null existem na variável (coluna) com o maior número de null? Responda como um único escalar.

# In[40]:


def q6():
    # Retorne aqui o resultado da questão 6.
    return (black_friday.shape[0] - black_friday['Product_Category_3'].dropna().shape[0])
q6()


# ## Questão 7
# 
# Qual o valor mais frequente (sem contar nulls) em `Product_Category_3`? Responda como um único escalar.

# In[41]:


def q7():
    # Retorne aqui o resultado da questão 7.
    return float(black_friday.Product_Category_3.mode())
q7()


# ## Questão 8
# 
# Qual a nova média da variável (coluna) `Purchase` após sua normalização? Responda como um único escalar.

# In[42]:


def q8():
    # Retorne aqui o resultado da questão 8.
    return float(Purchase_Normalized.mean())
q8()


# ## Questão 9
# 
# Quantas ocorrências entre -1 e 1 inclusive existem da variáel `Purchase` após sua padronização? Responda como um único escalar.

# In[43]:


def q9():
    # Retorne aqui o resultado da questão 9.
    return int(Purchase_Standardization['Purchase_STD'].between(-1,1).sum())
q9()


# ## Questão 10
# 
# Podemos afirmar que se uma observação é null em `Product_Category_2` ela também o é em `Product_Category_3`? Responda com um bool (`True`, `False`).

# In[44]:


def q10():
    # Retorne aqui o resultado da questão 10.
    return bool(len(black_friday.query("Product_Category_2 == 'NaN' & Product_Category_3 == 'NaN'")))
q10()

