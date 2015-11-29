import numpy as np

raw_data = open ('data.txt')
dataset = np.loadtxt(raw_data,delimiter=",")
matr = dataset[:,:]
def swap(index1, index2, iterable):
    for x in iterable:
        x[index1],x[index2]=x[index2],x[index1]
swap(0,6,matr)
print matr
X_all = matr[:,1:]
        # + matr[:, 7:13] + matr[:, 13:19] + matr[:, 19:25]
#print matr
y_all = dataset [:,0]
X_train = matr [0:150,1:]
    # matr[0:150,0:6] + matr[0:150, 7:13] + matr[0:150, 13:19] + matr[0:150, 19:25]
#print X_train
y_train = dataset [0:150,0]
#print y_train
X_test = dataset[150:194,1:]
    # dataset[150:194,0:6] + matr[150:194, 7:13] + matr[150:194, 13:19] + matr[150:194, 19:25]
y_test = dataset [150:194,0]
    # dataset[150:194,6]

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_all = scaler.fit_transform(X_all)

from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(X_train, y_train)
expected = y_test
predicted = model.predict(X_test)
print expected
print predicted

# Import the random forest package
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)
predicted = model.predict(X_test)
expected = y_test
print expected
print predicted


from sklearn.ensemble import GradientBoostingClassifier
model = GradientBoostingClassifier(loss='deviance', learning_rate=0.1, n_estimators=100, subsample=1.0, min_samples_split=2, min_samples_leaf=1,
                                   min_weight_fraction_leaf=0.0, max_depth=1, init=None, random_state=None, max_features=None, verbose=0,
                                   max_leaf_nodes=None, warm_start=False)
model.fit(X_train, y_train)
expected = y_test
predicted = model.predict(X_test)
print expected
print predicted


from sklearn.cross_validation import KFold

def run_cv(X,y,clf_class,**kwargs):
    kf = KFold(len(y),n_folds=10,shuffle=True)
    y_pred = y.copy()

    for train_index, test_index in kf:
        X_train, X_test = X[train_index], X[test_index]
        y_train = y[train_index]
        clf = clf_class(**kwargs)
        clf.fit(X_train,y_train)
        y_pred[test_index] = clf.predict(X_test)
    return y_pred

def accuracy(y_true,y_pred):
    return np.mean(y_true == y_pred)
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.naive_bayes import GaussianNB as GNB
from sklearn.ensemble import GradientBoostingClassifier as GBC
print "RandomForestClassifier:"
print "%.3f" % accuracy(y_all, run_cv(X_all,y_all,RFC))
print "GaussianNB:"
print "%.3f" % accuracy(y_all, run_cv(X_all,y_all,GNB))
print "GradientBoostingClassifier:"
print "%.3f" % accuracy(y_all, run_cv(X_all,y_all,GBC))