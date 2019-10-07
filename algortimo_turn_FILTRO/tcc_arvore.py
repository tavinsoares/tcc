import csv
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif,  SelectPercentile, SelectFpr, SelectFwe, SelectFdr
from sklearn.model_selection import RepeatedStratifiedKFold, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
import numpy as np
from inspect import ismethod
from sklearn import tree


base = ['ZP-PreFlop.dat', 'ZP-Flop.dat', 'ZP-River.dat', 'ZP-Turn.dat' ,'ZP-PostFlop.dat']
features_names = np.array(['POSITION',  'EHS', 'TOTAL_POT', 'POT_ODDS', 'BOARD_SUIT', 'BOARD_CARDS', 'BOARD_CONNECT', 'PREV_ROUND_ACTION', 'PREVIOUS_ACTION', 'BET_VILLAN', 'AGG', 'IP_VS', 'OP_VS'])

parameters = [
   ('kbest' , 'kbest__k', [3, 4, 5, 6, 7,8,9]),
   ('perc' , 'perc__percentile', [30, 40, 50, 60, 70]), 
   ('knn' ,'knn__n_neighbors' , [1, 2,3,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]),
   ('tree' ,'tree__min_samples_leaf', [2, 5, 10, 0]),
   ('tree' ,'tree__min_samples_split', [2, 5, 10, 0]),
   ('fpr', 'fpr__alpha', [0.02, 0.03, 0.04, 0.05, 0.06, 0.07]),
   ('fwe', 'fwe__alpha', [0.02, 0.03, 0.04, 0.05, 0.06, 0.07]),
   ('fdr', 'fdr__alpha', [0.02, 0.03, 0.04, 0.05, 0.06, 0.07]),
   ('variance', 'variance__threshold', [0.0, 0.01, 0.02, 0.03]),
   ('rf', 'rf__n_estimators', [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]),
   ('regression', 'regression__penalty', ['l1', 'l2']),
   ('regression', 'regression__C', [4, 16, 64, 256, 1024, 4096,16384]),
   ('mlp', 'mlp__learning_rate_init', [0.01, 0.05, 0.1, 0.2, 0.3]),
   ('mlp', 'mlp__hidden_layer_sizes', [10, 20, 50, 100, 500, 1000]),
   ('mlp', 'mlp__max_iter', [500])
]


listPipes = [

Pipeline([
	('variance', VarianceThreshold()),
	('tree', DecisionTreeClassifier())
	]),
	
	Pipeline([
    ('fwe', SelectFwe()),
	('tree' , DecisionTreeClassifier())
	]),

	Pipeline([
    ('fpr', SelectFpr()),
	('tree' , DecisionTreeClassifier())
	]),
	
	Pipeline([
	('fdr', SelectFdr()),
	('tree', DecisionTreeClassifier())
	]),

	Pipeline([
    ('kbest', SelectKBest()),
	('tree' , DecisionTreeClassifier())
	]),

	Pipeline([
    ('perc', SelectPercentile()),
	('tree' , DecisionTreeClassifier())
	])
]

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

def gravaResultados(dados, atrib, t):

	for i in dados:
		fileWrite = open(i[0]+i[1]+"_tree.txt","a")
		for j in i:
			fileWrite.write(str(j)+" ")
		fileWrite.write("\n")
	fileWrite.write(str(t))

	for k in atrib:
		fileWrite.write("\n"+str(features_names[k == True])+" "+str(len(features_names[k == True]))+"\n")
    
    
	fileWrite.close()
	
def pegaParametros(c, f):
	parms = {}

	for p in parameters:
		if(p[0] == c):
			parms[p[1]] = p[2]
		if(p[0] == f):
			parms[p[1]] = p[2]

	return parms


def pegaAtributos(name_steps):
	
	keys = name_steps.keys()
	if hasattr(name_steps[keys[0]], "get_support") and ismethod(getattr(name_steps[keys[0]], "get_support")):
		return name_steps[keys[0]].get_support()
	else:
		return name_steps[keys[1]].get_support()	

def plotTree(name_steps):
    keys = name_steps.keys()

    if hasattr(name_steps[keys[0]], "tree_"):
        atrib = name_steps[keys[1]].get_support()
        t = tree.export_graphviz(name_steps[keys[0]], impurity=False, out_file=str(f)+str(c)+"tree.dot", label='none', class_names=['1.0', '2.0', '3.0', '4.0', '5.0'], feature_names=[np.asarray(features_names[x]) for x in range(len(atrib)) if np.asarray(atrib[x]) == True])
        return t
    else:
        atrib = name_steps[keys[0]].get_support()
        t = tree.export_graphviz(name_steps[keys[1]], impurity=False, out_file=str(f)+str(c)+"tree.dot", label='none',  class_names=['1.0', '2.0', '3.0', '4.0', '5.0'], feature_names=[np.asarray(features_names[x]) for x in range(len(atrib)) if np.asarray(atrib[x]) == True])
        return t

def executa(pipe, parms):
	dados = []
	accuracy = []
	atrib = []
	outer_rskf = RepeatedStratifiedKFold(n_splits=10, n_repeats=1)
	inner_rskf = RepeatedStratifiedKFold(n_splits=3, n_repeats=1)
	
	for train_ids, test_ids in outer_rskf.split(X, y):
		if('tree__min_samples_leaf' in parms):
			parms['tree__min_samples_leaf'][-1] = ((len(train_ids)*1)/100)
		
		if('tree__min_samples_split' in parms):
			parms['tree__min_samples_split'][-1] = ((len(train_ids)*1)/100)
		
		grid = GridSearchCV(pipe, parms, verbose=1,n_jobs=4, cv=inner_rskf)
		grid.fit(X[train_ids].astype(str), y[train_ids].astype(str))
		newlabels = grid.predict(X[test_ids].astype(str))
		atrib.append(pegaAtributos(grid.best_estimator_.named_steps))
        accuracy.append((np.mean(newlabels == y[test_ids].astype(str))))
       
	t = plotTree(grid.best_estimator_.named_steps)	
	dados.append((f, c, np.mean(np.asarray(accuracy)), np.std(np.asarray(accuracy))))
	accuracy = []

	return dados, atrib, t

X, y = baseDados()
X = np.array(X)
y = np.array(y)


for pipe in listPipes:
	f = pipe.named_steps.keys()[0]
	c = pipe.named_steps.keys()[1]
	parms = pegaParametros(f, c)
	dados, atrib, t = executa(pipe, parms)
	gravaResultados(dados, atrib, t)
	


		