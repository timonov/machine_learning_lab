import matplotlib.pyplot as plt
import pylab as pl
import numpy as np


def plot_classification_report(cm):
    labels = ['class 0', 'class 1', 'class 2', 'class 3', 'class 4', 'class 5', 'class 6', 'class 7', 'class 8']
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(cm)
    pl.title('Confusion matrix of the classifier')
    fig.colorbar(cax)
    ax.set_xticklabels([''] + labels)
    ax.set_yticklabels([''] + labels)
    pl.xlabel('Predicted')
    pl.ylabel('True')
    pl.show()


def default_plot_report(_x_train_steps, accuracy_score_bnb, accuracy_score_rfc, accuracy_score_gbc):
    pl.title("Dependence the method's accuracy of the value of training sample")
    pl.xlabel("_x_train_steps")
    pl.ylabel("accuracy_score")
    plt.plot(_x_train_steps, accuracy_score_bnb, label="NaiveBayes")
    plt.plot(_x_train_steps, accuracy_score_rfc, label="RandomForestClassifier")
    plt.plot(_x_train_steps, accuracy_score_gbc, label="GradientBoostingClassifier")
    pl.legend(loc="upper left")
    pl.show()


def plot_classification_report_for_each_method(cr, title, with_avg_total=False, cmap=plt.cm.Blues):

    lines = cr.split('\n')

    classes = []
    plot_mat = []
    for line in lines[2: (len(lines) - 3)]:
        # print(line)
        t = line.split()
        # print(t)
        classes.append(t[0])
        v = [float(x) for x in t[1: len(t) - 1]]
        # print(v)
        plot_mat.append(v)

    if with_avg_total:
        ave_total = lines[len(lines) - 1].split()
        classes.append('avg/total')
        v_ave_total = [float(x) for x in t[1:len(ave_total) - 1]]
        plot_mat.append(v_ave_total)
    plt.imshow(plot_mat, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    x_tick_marks = np.arange(3)
    y_tick_marks = np.arange(len(classes))
    plt.xticks(x_tick_marks, ['precision', 'recall', 'f1-score'], rotation=45)
    plt.yticks(y_tick_marks, classes)
    plt.tight_layout()
    plt.ylabel('Classes')
    plt.xlabel('Measures')
    plt.show()


def plot_diff_num_tree(n_estimators, accuracy_score_rfc, accuracy_score_gbc):
    pl.title("Dependence the method's accuracy of the value of n_estimators")
    pl.xlabel("n_estimators")
    pl.ylabel("accuracy_score")
    plt.plot(n_estimators, accuracy_score_rfc, label="RandomForestClassifier")
    plt.plot(n_estimators, accuracy_score_gbc, label="GradientBoostingClassifier")
    pl.legend(loc="lower right")
    pl.show()
