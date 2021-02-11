close all; clear variables; clc;

% importing the relevant data for the final model selected:

% dummy variable training set

dummy_train = readtable('dummy_train.csv');

% For some reason, when using readtable, i found categorical variables are reverted
% to char variables. Thus, the next step is changing relevant variables
% back to categorical class

dummy_train.WHO_classification = categorical(dummy_train.WHO_classification,...
    {'Insufficient_Weight','Normal_Weight','Overweight',...
    'Obesity_Type_I','Obesity_Type_II','Obesity_Type_III'});

% dummy variable test set

dummy_test = readtable('dummy_test.csv');

dummy_test.WHO_classification = categorical(dummy_test.WHO_classification,...
    {'Insufficient_Weight','Normal_Weight','Overweight', 'Obesity_Type_I','Obesity_Type_II','Obesity_Type_III'});

% splitting in to predictors and target variables

dummy_xTrain = dummy_train(:, 1:42);
dummy_yTrain = table(dummy_train.WHO_classification);
dummy_xTest = dummy_test(:, 1:42);
dummy_yTest = table(dummy_test.WHO_classification);

%% Preparing the data for modelling

% creating BMI feature

bmi_data = table([dummy_train.weight./(dummy_train.height.^2)]);
bmi_dummy_test = table([dummy_test.weight./(dummy_test.height.^2)]);

% switching out the height and weight features for the BMI feature

bmi_dummy_train = [bmi_data, dummy_train(:, 1), dummy_train(:, 4:end)];
bmi_dummy_test = [bmi_dummy_test, dummy_test(:, 1), dummy_test(:, 4:end)];

%% Modelling

rng(1)

% final_rf = fitcensemble(bmi_dummy_train, 'WHO_classification', 'Method', 'Bag',...
%     'OptimizeHyperparameters', {'NumLearningCycles', 'MinLeafSize', 'MaxNumSplits', 'SplitCriterion', 'NumVariablesToSample'},...
%     'HyperparameterOptimizationOptions', struct('Optimizer', 'gridsearch', 'MaxObjectiveEvaluations', 250));

% loading the final model (to save time on re-training)

load final_rf
%% Evaluating model performance

% create a 10-fold cross-validation of the model

cv = crossval(final_rf, 'Kfold', 10);

% calculate cross-val accuracy

crossvalAccuracy = (1-kfoldLoss(cv))*100;

% calculate performance measures on the test set

[preds, score] = predict(final_rf, bmi_dummy_test(:, 1:41));

% confusion matrix

results = confusionmat(table2array(bmi_dummy_test(:, 42)), preds)

results_sum = sum(sum(results));

% test accuracy percentage

testAccuracy = (sum(diag(results)) / results_sum) * 100;

% precision

precision = [];

for i = 1:size(results, 2)
    col = results(:, i);
    if sum(col) ~= 0
        tp = col(i);
        if tp ~= 0
            p = tp/sum(col);
            precision = [precision, p];
        else 
            p = 0;
            precision = [precision, p];
        end
    end
    
end

precision = sum(precision)/numel(precision);

% recall

recall = [];

for i = 1:size(results, 1)
    row = results(i, :);
    if sum(row) ~= 0
        tp = row(i);
        if tp ~= 0
            r = tp/sum(row);
            recall = [recall, r];
        else
            r = 0;
            recall = [recall, r];
        end
    end
end


recall = sum(recall)/numel(recall);

% F1 score

fscore = 2*((precision * recall)/(precision + recall));

% printing the relevant results

fprintf('cross-validation accuracy: %.2f%%', crossvalAccuracy)
fprintf('\n')
fprintf('test accuracy: %.2f%%', testAccuracy)
fprintf('\n')
fprintf('Precision: %.2f%', precision)
fprintf('\n')
fprintf('Recall: %.2f%', recall)
fprintf('\n')
fprintf('F1 score: %.2f%', fscore)

