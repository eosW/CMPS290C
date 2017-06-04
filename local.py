import numpy as np
import pandas as pd
from scipy import sparse
from sklearn import svm, datasets, model_selection, metrics, linear_model, multiclass

X, _ = datasets.load_svmlight_file("corpus.txt")
Y = pd.read_csv("rawtag.txt")
ids = pd.read_csv("ids.txt")

median = X.shape[0] // 2

X = sparse.hstack([X[:median, :], X[median:, :]]).tocsr()
Y = Y[Y.columns[1:]].values.astype(int)

X_train = X[ids.train.values.nonzero()[0], :]
X_test = X[(~ids.train).values.nonzero()[0], :]
Y_train = Y[ids.train, :]
Y_test = Y[~ids.train, :]

model = linear_model.LogisticRegression(class_weight="balanced", solver="lbfgs")
mmodel = multiclass.OneVsRestClassifier(model)

mmodel.fit(X_train, Y_train)

p = mmodel.predict_proba(X_test)

disagree = np.logical_and(p[:, 0] > 0.5, p[:, 0] > p[:, 1])
agree = np.logical_and(p[:, 1] > 0.5, p[:, 1] > p[:, 0])
emotion = np.logical_and(p[:, 2] > 0.5, p[:, 2] > p[:, 3])
fact = np.logical_and(p[:, 3] > 0.5, p[:, 3] > p[:, 2])
nasty = np.logical_and(p[:, 4] > 0.5, p[:, 4] > p[:, 5])
nice = np.logical_and(p[:, 5] > 0.5, p[:, 5] > p[:, 4])
attacking = np.logical_and(p[:, 6] > 0.5, p[:, 6] > p[:, 7])
respectful = np.logical_and(p[:, 7] > 0.5, p[:, 7] > p[:, 6])
sarcasm = p[:, 8] > 0.5

print(metrics.precision_recall_fscore_support(disagree, Y_test[:, 0], average='macro'))
print(metrics.precision_recall_fscore_support(agree, Y_test[:, 1], average='macro'))
print(metrics.precision_recall_fscore_support(emotion, Y_test[:, 2], average='macro'))
print(metrics.precision_recall_fscore_support(fact, Y_test[:, 3], average='macro'))
print(metrics.precision_recall_fscore_support(nasty, Y_test[:, 4], average='macro'))
print(metrics.precision_recall_fscore_support(nice, Y_test[:, 5], average='macro'))
print(metrics.precision_recall_fscore_support(attacking, Y_test[:, 6], average='macro'))
print(metrics.precision_recall_fscore_support(respectful, Y_test[:, 7], average='macro'))
print(metrics.precision_recall_fscore_support(sarcasm, Y_test[:, 8], average='macro'))

tag = pd.DataFrame(
    {'id': ids[~ids.train].id, 'disagree': disagree, 'agree': agree, 'emotion': emotion, 'fact': fact, 'nasty': nasty,
     'nice': nice, 'attacking': attacking, 'respectful': respectful, 'sarcasm': sarcasm})

tag = tag.melt(id_vars=['id'],
               value_vars=['disagree', 'agree', 'emotion', 'fact', 'nasty', 'nice', 'attacking', 'respectful','sarcasm'])
tag = tag[tag.value][['id', 'variable']]
tag.to_csv('PSL/SVMTagging.txt', sep="\t", index=False, header=False)
