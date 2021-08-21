import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('cr_loan2.csv', index_col=  0)
df = pd.DataFrame(df)

print(df.shape)
print(df.head())
print(df.columns)
print(df.info())

letra_int = {
  "RENT" : "1",
  "OWN" : "2",
  "MORTGAGE" : 3,
}

intent = {
  "PERSONAL" : "1",
  "EDUCATION" : "2",
  "MEDICAL" : "3",
  "HOMEIMPROVEMENT" : "4"
}

grade = {
  "A" : "1",
  "B" : "2",
  "C" : "3",
  "D" : "4"
}


propiedad_casa = df['person_home_ownership'].map(letra_int)
intencion_prestamo = df['loan_intent'].map(intent)
calificacion = df['loan_grade'].map(grade)

df['person_home_ownership'] = propiedad_casa
df['loan_intent'] = intencion_prestamo
df['loan_grade'] = calificacion

print(df.head())

df_2 = df.drop(columns=['cb_person_default_on_file'])
print(df_2.columns)

print(pd.crosstab(df_2['person_home_ownership'],[df_2['loan_status'],df_2['loan_grade']]))

print(pd.crosstab(df_2['person_home_ownership'],df_2['loan_status'],values=df_2['loan_percent_income'],aggfunc='mean'))

df_2.boxplot(column=['loan_percent_income'],by='loan_status')
plt.title('Average Percent Income by Loan Status')
plt.suptitle('')
plt.show()

n,bins,patches=plt.hist(x=df_2['loan_amnt'],bins='auto',color='blue',alpha=0.7,rwidth=0.85)
plt.xlabel('Loan Amount')
plt.show()

plt.scatter(df_2['person_emp_length'],df_2['person_income'],c='blue',alpha=0.5)
plt.xlabel('Person Employment Length by Months')
plt.ylabel('Person Income')
plt.show()

sns.set_style('ticks')
sns.color_palette('Blues',as_cmap=True)
ax=plt.subplots(figsize=(10,7))

sns.boxplot(df_2['person_income'])

sns.color_palette('Blues',as_cmap=True)

df_index=df_2.reset_index()

plt.figure(figsize=(8,6))
ax=sns.heatmap(df_index.corr(method='spearman'),vmin=1,vmax=1,annot=True,cmap='YlGnBu',linewidths=.5)

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

print(df_index.dtypes)

df_index=df_index.dropna(axis=0).copy()

df_index['person_home_ownership']=df_index['person_home_ownership'].astype(int)
df_index['loan_intent']=df_index['loan_intent'].astype(int)
df_index['loan_grade']=df_index['loan_grade'].astype(int)

y=df_index['loan_status']
X=df_index[["person_age", 'person_income', 'person_home_ownership', 'person_emp_length',
       'loan_intent', 'loan_grade', 'loan_amnt', 'loan_int_rate', 'loan_percent_income',
       'cb_person_cred_hist_length']]
       
X_test,X_train,y_test,y_train=train_test_split(X,y,test_size=0.3)

from sklearn.metrics import confusion_matrix

def calcularAccuracy(TP, TN, FP, FN):
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    accuracy = accuracy * 100
    return accuracy
def calcularSensibilidad(TP, TN, FP, FN):
    sensibilidad = TP / (TP + FN)
    sensibilidad = sensibilidad * 100
    return sensibilidad
def calcularEspecificidad(TP, TN, FP, FN):
    especificidad = TN / (TN + FP)
    especificidad = especificidad * 100
    return especificidad

def evaluar(y_test, y_pred):
    resultado = confusion_matrix(y_test, y_pred)
    print(resultado)
    (TN, FP, FN, TP) = resultado.ravel()
    print("True positives: "+str(TP))
    print("True negatives: "+str(TN))
    print("False positives: "+str(FP))
    print("False negative: "+str(FN))

    acc = calcularAccuracy(TP, TN, FP, FN)
    sen = calcularSensibilidad(TP, TN, FP, FN)
    spec = calcularEspecificidad(TP, TN, FP, FN)
    print("Precision:"+str(acc)+"%")
    print("Sensibilidad:"+str(sen)+"%")
    print("Especificidad:"+str(spec)+"%")

from sklearn.naive_bayes import GaussianNB

NaiveBayes = GaussianNB()# Inicializa el modelo
NaiveBayes.fit(X_train, y_train) # Ajusta el modelo
y_pred_nb = NaiveBayes.predict(X_test) # Genera la predicción 

print(evaluar(y_test, y_pred_nb))

from sklearn.svm import SVC #Support Vector Classifier

SupportVectorMachine =SVC() # Inicializa el modelo
SupportVectorMachine.fit(X_train, y_train) # Ajusta el modelo
y_pred_svm =  SupportVectorMachine.predict(X_test)# Genera la predicción 

print(evaluar(y_test, y_pred_svm))

from sklearn.neural_network import MLPClassifier

NeuralNetwork =  MLPClassifier(hidden_layer_sizes=(10,6),# Inicializa el modelo
                            max_iter=2000000,
                            activation ='tanh',
                            tol= 1e-8) #Ajusta el modelo
NeuralNetwork.fit(X_train,y_train)

y_pred_nn = NeuralNetwork.predict(X_test)# Genera la predicción

print(evaluar(y_test, y_pred_nn))


from sklearn.ensemble import RandomForestClassifier

RandomForest =RandomForestClassifier(n_estimators = 800) # Inicializa el modelo
RandomForest.fit(X_train, y_train)# Ajusta el modelo
y_pred_rfc =  RandomForest.predict(X_test)#Genera la predicción

print(evaluar(y_test, y_pred_rfc))

def acc(y_test, y_pred):
    resultado = confusion_matrix(y_test, y_pred)
    (TN, FP, FN, TP) = resultado.ravel()
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    accuracy = accuracy * 100
    return accuracy

def sec(y_test, y_pred):
    resultado = confusion_matrix(y_test, y_pred)
    (TN, FP, FN, TP) = resultado.ravel()
    sensibilidad = TP / (TP + FN)
    sensibilidad = sensibilidad * 100
    return sensibilidad

def esp(y_test, y_pred):
    resultado = confusion_matrix(y_test, y_pred)
    (TN, FP, FN, TP) = resultado.ravel()
    especificidad = TN / (TN + FP)
    especificidad = especificidad * 100
    return especificidad

from tabulate import tabulate

table = [['Naive Bayes', acc(y_test, y_pred_nb), sec(y_test, y_pred_nb), esp(y_test, y_pred_nb)],
        ['Support Vector Machine', acc(y_test, y_pred_svm), sec(y_test, y_pred_svm), esp(y_test, y_pred_svm)],
        ['Artificial Neural Network', acc(y_test, y_pred_nn), sec(y_test, y_pred_nn), esp(y_test, y_pred_nn)],
        ['Random Forest', acc(y_test, y_pred_rfc), sec(y_test, y_pred_rfc), esp(y_test, y_pred_rfc)]]

print(tabulate(table,
               headers = ['Método', 'Accuracy', 'Sensibilidad', 'Especificidad'],
               stralign = 'right',
               floatfmt = '.2f',
               tablefmt = 'simple'))
