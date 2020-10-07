# Google_Quest

## Submissions are evaluated on the mean column-wise Spearman's correlation coefficient. The Spearman's rank correlation is computed for each target column, and the mean of these values is calculated for the submission score.
Submission File
For each qa_id in the test set, you must predict a probability for each target variable. The predictions should be in the range [0,1]. The file should contain a header and have the following format:
qa_id,question_asker_intent_understanding,...,answer_well_written 6,0.0,...,0.5 8,0.5,...,0.1 18,1.0,...,0.0 etc.
