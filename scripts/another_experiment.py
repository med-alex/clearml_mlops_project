import os
import matplotlib.pyplot as pyplot
import matplotlib.pyplot as plt
import itertools
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, precision_score, recall_score, f1_score
from sklearn.metrics import log_loss
from clearml import Task
task = Task.init(project_name="TPS", task_name="blending")


path = Path.cwd() / 'data'

submission = pd.read_csv(path / 'sample_submission.csv', index_col='id')
labels = pd.read_csv(path / 'train_labels.csv', index_col='id')

# the ids of the submission rows (useful later)
sub_ids = submission.index

# the ids of the labeled rows (useful later)
gt_ids = labels.index 

# list of files in the submission folder
subs = sorted(os.listdir(path / 'submission_files'))

# number of file to consider
nb_files = 3000

train = []
for i in range(0, nb_files):
    train.append(pd.read_csv(path / 'submission_files' / subs[i], index_col='id')['pred'].values)

train = np.stack(train).T

meta_logistic = LogisticRegression(solver='liblinear')
meta_logistic.fit(train[gt_ids], labels['label'].values)
preds_linear = meta_logistic.predict_proba(train)[:,1] # we take predictions for positive values

# Evaluate the log-loss on the train set.
score = log_loss(labels, preds_linear[gt_ids])

print(f'score: {score:.10f}')

data = {'id':  sub_ids,
        'pred': preds_linear[sub_ids]}

submission = pd.DataFrame(data)
submission = submission.set_index('id')
submission.to_csv(Path.cwd() / 'results' / 'submission.csv')

task.upload_artifact(name="Score", artifact_object={'score': score})
task.get_logger().report_table ("submission csv",
                                series="submission.csv",
                                table_plot=submission)

# probability above 
y_pred = preds_linear[gt_ids] > .5

y = labels['label'].values
cf_mat = confusion_matrix(y, y_pred)

# Define the function to plot the confusion matrix
def plot_confusion_matrix(cm, classes, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    
# Function to print matrix
def metric_scores(y_reference, y_prediction):
    precision = precision_score(y_reference, y_prediction)
    recall = recall_score(y_reference, y_prediction)
    F1_score = f1_score(y_reference, y_prediction)
    print('precision: '+ str(precision.round(3)))
    print('recall: '+ str(recall.round(3)))
    print('F1_score: '+ str(F1_score.round(3)))
    return(precision, recall, F1_score)

#plot confusion matrix
cm_plot=plot_confusion_matrix(cf_mat, ['true','false'])

lr_auc = roc_auc_score(y, y_pred)
# summarize scores
print('Logistic: ROC AUC=%.3f' % (lr_auc))
# calculate roc curves
lr_fpr, lr_tpr, _ = roc_curve(y, y_pred)

# plot the roc curve for the model
pyplot.plot(lr_fpr, lr_tpr, marker='.', label='Logistic')
# axis labels
pyplot.xlabel('False Positive Rate')
pyplot.ylabel('True Positive Rate')
# show the legend
pyplot.legend()
pyplot.grid()
# show the plot
pyplot.show()
