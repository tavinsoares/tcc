import csv 
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.feature_selection import RFECV
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score as acc
from sklearn.svm import SVC

def baseDados():
	data = []

	with open('ZP-Turn.dat') as csvfile:
	    arq = csv.reader(csvfile, delimiter=' ')

	    for l in arq:
	        data.append(l)

	classes_data = []

	for i in data:
	    classes_data.append(i[len(i)-1])
	    del(i[len(i)-1])

	return data, classes_data

def gravaResultados(name, dados, atrib):
    fileWrite = open("wrapper"+name+".txt","a")
	
    fileWrite.write("Acuracia: "+str(dados[0])+"\n"+
                    "N de atributos selecionados: "+str(dados[1])+"\n")

    fileWrite.write("Atributos selecionados: \n"+str(features_names[atrib == True])+"\n")
	
    fileWrite.close()

def Parametros(name_c):
    params = {
    'min_features_to_select' : [10, 9, 8, 7 ,6, 5 ],
    }
    for x in list_parms:
        if name_c in x:
            params[x[1]] = x[2]      

    return params

def feature_selection(c):
    dados = []
    rfe = RFECV(estimator=c[1], cv=StratifiedKFold(n_splits=3))
    grid = GridSearchCV(rfe, Parametros(c[0]), verbose=1,n_jobs=4)
    grid.fit(x_train.astype(float), y_train.astype(float))
    newlabels = grid.predict(x_test.astype(float))

    dados.append(np.mean(newlabels == y_test.astype(float)))
    dados.append(grid.best_estimator_.n_features_ )
    gravaResultados(c[0], dados, grid.best_estimator_.support_)

features_names = np.array(['POSITION',  'EHS', 'TOTAL_POT', 'POT_ODDS', 'BOARD_SUIT', 'BOARD_CARDS', 'BOARD_CONNECT', 'PREV_ROUND_ACTION', 'PREVIOUS_ACTION', 'BET_VILLAN', 'AGG', 'IP_VS', 'OP_VS'])

x, y = baseDados()

x = np.array(x)
y = np.array(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=42)

list_classifier = [
    ["LogisticRegression", LogisticRegression()],
    ["RandomFlorest", RandomForestClassifier()],
    ["DecisionTree", DecisionTreeClassifier()],
    ["SVM", SVC(kernel='linear')]
]

list_parms = [
    ["LogisticRegression", "estimator__penalty", ['l1', 'l2']],
    ["LogisticRegression", "estimator__C", [4, 16, 64, 256, 1024, 4096,16384]],
    ["RandomFlorest", "estimator__n_estimators", [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]],
    ["DecisionTree", "estimator__min_samples_leaf", [2, 5, 10, len(x_train)/100]],
    ["DecisionTree", "estimator__min_samples_split", [2, 5, 10, len(x_train)/100]],
    ["SVM", "estimator__gamma", [2**4, 2**3, 2**2, 2, 1, 2**-1, 2**-2, 2**-3, 2**-4, 2**-5, 2**-6, 2**-7, 2**-8, 2**-9, 2**-10]],
    ["SVM", "estimator__C", [2**12, 2**11, 2**10, 2**9, 2**8, 2**7, 2**6, 2**5, 2**4, 2**3, 2**2, 2 , 1,2**-1,2**-2]]
]

for c in list_classifier:
    feature_selection(c)
