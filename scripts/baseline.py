import os
import pandas as pd 
from pathlib import Path
from sklearn.metrics import log_loss
from clearml import Task
task = Task. init(project_name="TPS", task_name="blending")

submission = pd.read_csv(Path.cwd() / 'data' / 'sample_submission.csv', index_col='id')
labels = pd.read_csv(Path.cwd() / 'data' / 'train_labels.csv', index_col='id')

# the ids of the submission rows (useful later)
sub_ids = submission.index
# the ids of the labeled rows (useful later)
gt_ids = labels.index

# list of files in the submission folder
# Notice the score of the labeled rows matches the file name
subs = sorted(os.listdir(Path.cwd() / 'data' / 'submission_files'))
s0 = pd.read_csv(Path.cwd() / 'data' / 'submission_files' / subs[0], index_col='id')
score = log_loss(labels, s0.loc[gt_ids])
# print(subs[0], f'{score:.10f)')
s1 = pd.read_csv(Path.cwd() / 'data' / 'submission_files' / subs[1], index_col='id')
score = log_loss(labels, s1.loc[gt_ids])
# print(subs[1], f'{score:.10f}')
blend = (s0 + s1) / 2
score = log_loss(labels, blend.loc[gt_ids])
task.upload_artifact(name="Score", artifact_object={'score': score})

#Log a result csv and write to output file task.get_logger)-report
task.get_logger().report_table ("Result csv",
                                series="blend.csv",
                                table_plot=blend.loc[sub_ids])
blend.loc[sub_ids].to_csv(Path.cwd() / 'blend.csv')
