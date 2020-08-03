import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
import json

# function to create dummy lista


def dummy(df, dummylist):
    for x in dummylist:
        dummies = pd.get_dummies(df[x], prefix=x, dummy_na=False)
        df = df.drop(x, axis=1)
        df = pd.concat([df, dummies], axis=1)
    return df


# load the training and testing data
train = 'testfiles/train.csv'
dt = pd.read_csv(train)
test = 'testfiles/test.csv'
datatest = pd.read_csv(test)


# deletes any rows which the score is null
dt = dt[dt['NU_NOTA_MT'].notna()]

# scores answer
y = dt['NU_NOTA_MT']

# saves the NU number fir the test data
num_de_insc = datatest.NU_INSCRICAO

# gets only the columns which the values are numerical
for i in dt.columns:
    if i not in list(datatest.columns):
        del dt[i]

with open('listavet.json', 'r') as lv:
    lista = json.load(lv)

# dropar = ['TP_DEPENDENCIA_ADM_ESC', 'IN_BAIXA_VISAO', 'IN_CEGUEIRA', 'IN_SURDEZ',
#           'IN_DISLEXIA', 'IN_DISCALCULIA', 'IN_SABATISTA', 'IN_GESTANTE', 'IN_IDOSO',
#           'TP_PRESENCA_CN', 'TP_PRESENCA_CH', 'TP_PRESENCA_LC', 'NU_INSCRICAO',
#           'SG_UF_RESIDENCIA', 'CO_PROVA_CN', 'CO_PROVA_CH', 'CO_PROVA_LC',
#           'CO_PROVA_MT', 'TP_STATUS_REDACAO', 'NU_NOTA_COMP1', 'NU_NOTA_COMP2', 'NU_NOTA_COMP3',
#           'NU_NOTA_COMP4', 'NU_NOTA_COMP5', 'Q001', 'Q002', 'Q006', 'Q024', 'Q025',
#           'Q026', 'Q027', 'Q047', 'NU_NOTA_CN', 'NU_NOTA_CH', 'NU_NOTA_LC',
#           'TP_LINGUA', 'NU_NOTA_REDACAO', 'TP_ANO_CONCLUIU', 'IN_TREINEIRO']
dropar = ['CO_PROVA_CN', 'CO_PROVA_CH', 'CO_PROVA_LC',
          'CO_PROVA_MT', 'TP_PRESENCA_CN', 'TP_PRESENCA_CH', 'TP_PRESENCA_LC',
          'NU_INSCRICAO', 'TP_STATUS_REDACAO', 'SG_UF_RESIDENCIA']
for i in dropar:
    dt = dt.drop(i, 1)
    datatest = datatest.drop(i, 1)

real_lista = []
for i in lista:
    if i not in dropar:
        real_lista.append(i)


A = dummy(dt, real_lista)
B = dummy(datatest, real_lista)

for i in A.columns:
    if i not in B.columns:
        A = A.drop(i, 1)

imp = SimpleImputer(strategy='constant', copy=False, fill_value=-1)
X = imp.fit_transform(A)
T = imp.fit_transform(B)

for i in range(len(X)):
    for j in range(len(X[i])):
        X[i, j] = round(X[i, j], 1)

for i in range(len(T)):
    for j in range(len(T[i])):
        T[i, j] = round(T[i, j], 1)

# linear regression
lin = LinearRegression()
lin.fit(X, y)

# predicts the result
t = lin.predict(T)
pd.DataFrame()

for i in range(len(t)):
    t[i] = round(t[i], 1)

# resposta = pd.DataFrame(t)
# resposta = pd.concat([resposta, num_de_insc], axis=1)
# resposta.to_csv("agorarola.csv")
