import matplotlib as mpl

mpl.use('TkAgg')
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from matplotlib import cm as colormap
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score

#
##
###
####
# On peut remplacer dans le code "RandomForest" par "ExtraTree" selon ce que l'on veut tester
####
###
##
#



# Parametres generaux
######################

filename = 'data/missmda_train.csv'
filename2 = 'data/missmda_test.csv'
delimiter = ','
random_state = 0

general_strategy = 'Regressor'
# general_strategy = 'Classifier'


# Quelques fonctions
######################

# Les deux fonctions ci_dessous sont equivalentes.

def error_rate(y_pred, y_test):
    '''
        Calcul du taux d'erreur
        version manuelle
    '''
    fp = (y_pred == "bad") & (y_test == "good")
    fn = (y_pred == "good") & (y_test == "bad")
    return float(sum(fp | fn)) / float(len(y_pred))


def error_rate_2(y_pred, y_test):
    '''
            Calcul du taux d'erreur
            version build-in
    '''
    return 1 - accuracy_score(y_pred, y_test)


def to_lvefbin(result_list, general_strategy='Regressor'):
    '''
        binarise (en "good" et "bad") une liste de float, si la strategie est de faire une regression 
    '''
    if general_strategy == 'Regressor':  # Regressor : il faut convertir lvef en lvefbin
        return pd.Series(data=["good" if y >= 40.0 else "bad" for y in result_list])
    else:  # Classifier
        return result_list

def load_data(file_uri, general_strategy='Regressor'):
    '''
        Chargement des donnees et passage en tableau disjonctif complet pour les variables categoriques
    '''
    data = pd.read_csv(file_uri, header=0, sep=delimiter, error_bad_lines=False)
    print(data.columns)

    columns_x = ['centre', 'country', 'gender', 'bmi', 'age', 'egfr', 'sbp', 'dbp',
                 'hr', 'copd', 'hypertension', 'previoushf', 'afib', 'cad']
    columns_x_categorical = ['centre', 'country', 'gender', 'copd', 'hypertension', 'previoushf', 'afib', 'cad']

    X = data[columns_x]

    for cat in columns_x_categorical:
        X = pd.concat([X, pd.get_dummies(X[cat], prefix='country')], axis=1)
        X.drop([cat], axis=1, inplace=True)
        X = X.iloc[:, :-1]  # la derniere colonne est inutile, elle est egale a 1 - somme(colonnes 0..N-1)

    if 'lvef' in data.columns:
        if general_strategy == 'Regressor':
            y = data.loc[:, 'lvef']
        else:
            y = data.loc[:, 'lvefbin']
    else:
        y = None

    return X, y


# Chargement des donnees et split en ensemble d'entrainement, test et validation.
#################################################################################

X, y = load_data(filename)
X2, _ = load_data(filename2)

print("\n")
print("Les variables explicatives (colonnes de X) sont : {}.".format(list(X.columns)))
print("La variable a expliquer est : {}.".format(y.name))

# data_test = pd.read_csv(filename_test, header=0, sep=delimiter, error_bad_lines=False)
# print(len(data_test))
# if len(data_test) != 987:
#    sys.exit("Missing values")

X_train, X_test, y_train, y_test = train_test_split(X, y.values.reshape(-1), test_size=0.20,
                                                          random_state=random_state)

# Basic training
##################

if general_strategy == 'Regressor':
    clf = RandomForestRegressor(n_estimators=100, random_state=random_state)
else:
    clf = RandomForestClassifier(n_estimators=100, random_state=random_state)

clf.fit(X_train, y_train)

y_train_bin_pred = to_lvefbin(clf.predict(X_train), general_strategy=general_strategy)
y_train_bin_real = to_lvefbin(y_train, general_strategy=general_strategy)
y_test_bin_pred = to_lvefbin(clf.predict(X_test), general_strategy=general_strategy)
y_test_bin_real = to_lvefbin(y_test, general_strategy=general_strategy)

print("Taux d'erreur (apprentissage) : {:.3f} %".format(error_rate_2(y_train_bin_pred, y_train_bin_real)))
print("Taux d'erreur (test)          : {:.3f} %".format(error_rate_2(y_test_bin_pred, y_test_bin_real)))


# Grid Search training
#######################

tuned_parameters = {'n_estimators': range(120, 141, 10),  # best : 130
                    'max_depth': range(12, 15, 1),  # best : 13/14
                    'max_features': range(19, 24, 2),  # best : 19/21
                    'min_samples_leaf': range(5, 8, 1)  # best : 4/5
                    }

if general_strategy == 'Regressor':
    clf = GridSearchCV(ExtraTreesRegressor(random_state=random_state), tuned_parameters, cv=5, n_jobs=-1, verbose=True)
    # clf = GridSearchCV(ExtraTreesRegressor(), tuned_parameters, cv=5, n_jobs=2, verbose=True)

else:
    clf = GridSearchCV(ExtraTreeClassifier(random_state=random_state), tuned_parameters, cv=5, n_jobs=-1,
                       verbose=True)

clf.fit(X_train, y_train)
# Cette commande trouve le meilleurs jeux de parametres et refit le classifier avec, sur la base X_train.

y_train_bin_pred = to_lvefbin(clf.predict(X_train), general_strategy=general_strategy)
y_train_bin_real = to_lvefbin(y_train, general_strategy=general_strategy)
y_test_bin_pred = to_lvefbin(clf.predict(X_test), general_strategy=general_strategy)
y_test_bin_real = to_lvefbin(y_test, general_strategy=general_strategy)

print("Taux d'erreur (apprentissage) : {:.3f} %".format(error_rate_2(y_train_bin_pred, y_train_bin_real)))
print("Taux d'erreur (test)          : {:.3f} %".format(error_rate_2(y_test_bin_pred, y_test_bin_real)))

print("Params")
print(clf.best_params_)

# On refitesur le classifier parametre optimalement sur toute la base
clf_final = ExtraTreesRegressor(random_state=random_state, n_estimators=130, max_depth=13,
                                max_features=20,min_samples_leaf=5)
clf_final.fit(X, y)
y2_pred = to_lvefbin(clf_final.predict(X2))
y2_pred.to_csv('data/predictions.csv',index=False)



#
# print("Optimise ExtraTreesClassifier")
# print("Score apprentissage  = %f" % clf.score(X_train, y_train))
# print("Score test = %f" % clf.score(X_test, y_test))
#
# print("Params")
# print(clf.best_params_)
# print("Best score")
# print(clf.best_score_)
# print("Variable importance")
# print(clf.best_estimator_.feature_importances_)
#
# # pred_test = clf.best_estimator_.predict(data_test)
# # print(pred_test)
# # df = pd.DataFrame(pred_test)
# # df.to_csv("python_extratrees.csv")
#
# max_depth = np.array(range(3, 11, 2))
# min_samples_split = np.array([range(3, 10, 2)])
# xx, yy = np.meshgrid(max_depth, min_samples_split)
#
# # affichage sous forme de wireframe des resultats des modeles evalues
# fig = plt.figure()
# ax = fig.gca(projection='3d')
# Z = clf.cv_results_['mean_test_score'].reshape(xx.shape)
# # ax = fig.add_subplot(111, projection='3d')
# ax.plot_surface(xx, yy, Z, cmap=colormap.coolwarm)
# ax.set_xlabel("Profondeur")
# ax.set_ylabel("Nombre d'estimateurs")
# ax.set_zlabel("Score moyen")
# plt.show()
#
# fig = plt.figure()
# plt.plot(range(50, 1000, 50), clf.cv_results_['mean_test_score'])
# plt.show()
