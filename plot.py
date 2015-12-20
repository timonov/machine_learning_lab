import matplotlib.pyplot as plt
import pylab as pl


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
