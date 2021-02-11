close all; clear variables; clc;

% Reading in the 3 sepaarate data sets from pre-processing

data = readtable('prepped_raw_training_data.csv');

ne = readtable('numeric_encoding_train.csv');

dummy = readtable('dummy_train.csv');

% For some reason, when using readtable, categorical variables are reverted
% to char variables. Thus, the next step is changing relevant variables
% back to categorical class

data.gender = categorical(data.gender);
data.family_history_obesity = categorical(data.family_history_obesity);
data.fast_food_intake = categorical(data.fast_food_intake);
data.vegetable_consumption_freq = categorical(data.vegetable_consumption_freq);
data.number_of_meals_daily = categorical(data.number_of_meals_daily);
data.snacking_freq = categorical(data.snacking_freq);
data.smoker = categorical(data.smoker);
data.liquid_intake_daily = categorical(data.liquid_intake_daily);
data.calorie_counter = categorical(data.calorie_counter); 
data.physical_activity = categorical(data.physical_activity);
data.technology_usage = categorical(data.technology_usage);
data.alcohol_weekly = categorical(data.alcohol_weekly);
data.main_transportation = categorical(data.main_transportation);
data.WHO_classification = categorical(data.WHO_classification);

ne.main_transportation = categorical(ne.main_transportation);
ne.WHO_classification = categorical(ne.WHO_classification);

dummy.WHO_classification = categorical(dummy.WHO_classification);


% importing test data for each

test = readtable('prepped_raw_test_data');

% making all categorical variables categorical (see above)

test.gender = categorical(test.gender);
test.family_history_obesity = categorical(test.family_history_obesity);
test.fast_food_intake = categorical(test.fast_food_intake);
test.vegetable_consumption_freq = categorical(test.vegetable_consumption_freq);
test.number_of_meals_daily = categorical(test.number_of_meals_daily);
test.snacking_freq = categorical(test.snacking_freq);
test.smoker = categorical(test.smoker);
test.liquid_intake_daily = categorical(test.liquid_intake_daily);
test.calorie_counter = categorical(test.calorie_counter); 
test.physical_activity = categorical(test.physical_activity);
test.technology_usage = categorical(test.technology_usage);
test.alcohol_weekly = categorical(test.alcohol_weekly);
test.main_transportation = categorical(test.main_transportation);
test.WHO_classification = categorical(test.WHO_classification);

% numeric_encoding test set

ne_test = readtable('numeric_encoding_test.csv');

% making all categorical variables categorical (see above)

ne_test.WHO_classification = categorical(ne_test.WHO_classification);
ne_test.main_transportation = categorical(ne_test.main_transportation);

% dummy test set

dummy_test = readtable('dummy_test.csv');

dummy_test.WHO_classification = categorical(dummy_test.WHO_classification);

%% Dividing in to independent and target variables

% data
xTrain = data(:, 1:16);
yTrain = table(data.WHO_classification);
xTest = test(:, 1:16);
yTest = table(test.WHO_classification);

%numerically encoded data
ne_xTrain = ne(:, 1:16);
ne_yTrain = table(ne.WHO_classification);
ne_xTest = ne_test(:, 1:16);
ne_yTest = table(ne_test.WHO_classification);

% dummy variable data
dummy_xTrain = dummy(:, 1:42);
dummy_yTrain = table(dummy.WHO_classification);
dummy_xTest = dummy_test(:, 1:42);
dummy_yTest = table(dummy_test.WHO_classification);
%% Optimizing hyperparameters using gridsearch
% data

rng(1)

nb = fitcnb(data, 'WHO_classification', 'OptimizeHyperparameters', 'all',...
    'HyperparameterOptimizationOptions', struct('Optimizer', 'gridsearch', 'MaxObjectiveEvaluations', 250));

% create a 10-fold cross-validation of the model

cv = crossval(nb, 'Kfold', 10);

% calculate cross-val accuracy

crossvalAccuracy = (1-kfoldLoss(cv))*100

% calculate test set accuracy

[preds, score] = predict(nb, xTest);

results = confusionmat(table2array(yTest), preds);

results_sum = sum(sum(results));

testAccuracy = (sum(diag(results)) / results_sum) * 100

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
fprintf('F1 score: %.2f%', fscore)
 
 

%% numerically encoded data

rng(1)

ne_nb = fitcnb(ne, 'WHO_classification', 'OptimizeHyperparameters', 'all',...
    'HyperparameterOptimizationOptions', struct('Optimizer', 'gridsearch', 'MaxObjectiveEvaluations', 250));

% create a 10-fold cross-validation of the model

cv = crossval(ne_nb, 'Kfold', 10);

% calculate cross-val accuracy

crossvalAccuracy = (1-kfoldLoss(cv))*100

% calculate test set accuracy

[preds, score] = predict(ne_nb, ne_xTest);

results = confusionmat(table2array(ne_yTest), preds);

results_sum = sum(sum(results));

testAccuracy = (sum(diag(results)) / results_sum) * 100

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
fprintf('F1 score: %.2f%', fscore)
 
 


%% Dummy variable data

rng(1)

dummy_nb = fitcnb(dummy, 'WHO_classification', 'OptimizeHyperparameters', 'all',...
    'HyperparameterOptimizationOptions', struct('Optimizer', 'gridsearch', 'MaxObjectiveEvaluations', 250));


% create a 10-fold cross-validation of the model

cv = crossval(dummy_nb, 'Kfold', 10);

% calculate cross-val accuracy

crossvalAccuracy = (1-kfoldLoss(cv))*100

% calculate test set accuracy

[preds, score] = predict(dummy_nb, dummy_xTest);

results = confusionmat(table2array(dummy_yTest), preds);

results_sum = sum(sum(results));

testAccuracy = (sum(diag(results)) / results_sum) * 100

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
fprintf('F1 score: %.2f%', fscore)

%% Principal Component Analysis (PCA)

% extracting numeric values, and preparing for PCA

numeric = table2array(ne(:, 1:15));

% running PCA

[pcs,scrs,~,~,pctExp] = pca(numeric);

test_numeric = table2array(ne_test(:, 1:15));

[~, tscrs, ~, ~, ~] = pca(test_numeric);

% visualising PCA

pareto(pctExp)

%% Running Naive Bayes model with 2 most influential variables in terms of PCA

rng(1)


data_reduced = array2table(scrs(:, 1:2));
test_reduced = array2table(tscrs(:, 1:2));


data_reduced(:, 3) = table(data.WHO_classification, 'VariableNames', {'WHO_classification'});
test_redcued(:, 3) = table(test.WHO_classification, 'VariableNames', {'WHO_classification'});

pca_nb = fitcnb(data_reduced, 'Var3', 'OptimizeHyperparameters', 'all',...
    'HyperparameterOptimizationOptions', struct('Optimizer', 'gridsearch', 'MaxObjectiveEvaluations', 250));

cv = crossval(pca_nb, 'Kfold', 10);

% calculate cross-val accuracy

crossvalAccuracy = (1-kfoldLoss(cv))*100;

% calculate test set accuracy

[preds, score] = predict(pca_nb, test_reduced(:, 1:end-1));

results = confusionmat(table2array(test_reduced(:, end)), preds);

results_sum = sum(sum(results));

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
fprintf('F1 score: %.2f%', fscore)
 


%% feature engineering

rng(1)

bmi = table([data.weight./(data.height.^2)]);
bmi_data_test = table([test.weight./(test.height.^2)]);



bmi_data = [bmi, data(:, 1:2), data(:, 5:end)];
bmi_data_test = [bmi_data_test, test(:, 1:2), test(:, 5:end)];


bmi_data_nb = fitcnb(bmi_data, 'WHO_classification', 'OptimizeHyperparameters', 'all',...
    'HyperparameterOptimizationOptions', struct('Optimizer', 'gridsearch', 'MaxObjectiveEvaluations', 250));


% create a 10-fold cross-validation of the model

cv = crossval(bmi_data_nb, 'Kfold', 10);

% calculate cross-val accuracy

crossvalAccuracy = (1-kfoldLoss(cv))*100;

% calculate performance measures on the test set

[preds, score] = predict(bmi_data_nb, bmi_data_test(:, 1:15));

% confusion matrix

results = confusionmat(table2array(bmi_data_test(:, 16)), preds);

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
fprintf('F1 score: %.2f%', fscore)

save bmi_data_nb

%% numerically encoded data with the bmi feature engineering

rng(1)

bmi_data = table([ne_xTrain.weight./(ne_xTrain.height.^2)]);
bmi_ne_test = table([ne_xTest.weight./(ne_xTest.height.^2)]);



bmi_ne_train = [bmi_data, ne(:, 1:2), ne(:, 5:end)];
bmi_ne_test = [bmi_ne_test, ne_test(:, 1:2), ne_test(:, 5:end)];


bmi_ne_nb = fitcnb(bmi_ne_train, 'WHO_classification', 'OptimizeHyperparameters', 'all',...
    'HyperparameterOptimizationOptions', struct('Optimizer', 'gridsearch', 'MaxObjectiveEvaluations', 250));


% create a 10-fold cross-validation of the model

cv = crossval(bmi_ne_nb, 'Kfold', 10);

% calculate cross-val accuracy

crossvalAccuracy = (1-kfoldLoss(cv))*100;

% calculate performance measures on the test set

[preds, score] = predict(bmi_ne_nb, bmi_ne_test(:, 1:15));

% confusion matrix

results = confusionmat(table2array(bmi_ne_test(:, 16)), preds);

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
fprintf('F1 score: %.2f%', fscore)

%% and finally, the bmi feature engineering is tested on the dummy variable
% version of the data

rng(1)

bmi_data = table([dummy.weight./(dummy.height.^2)]);
bmi_dummy_test = table([dummy_test.weight./(dummy_test.height.^2)]);



bmi_dummy_train = [bmi_data, dummy(:, 1:2), dummy(:, 5:end)];
bmi_dummy_test = [bmi_dummy_test, dummy_test(:, 1:2), dummy_test(:, 5:end)];


bmi_dummy_nb = fitcnb(bmi_dummy_train, 'WHO_classification',...
    'OptimizeHyperparameters', 'all',...
    'HyperparameterOptimizationOptions', struct('Optimizer', 'gridsearch', 'MaxObjectiveEvaluations', 250));

% create a 10-fold cross-validation of the model

cv = crossval(bmi_dummy_nb, 'Kfold', 10);

% calculate cross-val accuracy

crossvalAccuracy = (1-kfoldLoss(cv))*100;

% calculate performance measures on the test set

[preds, score] = predict(bmi_dummy_nb, bmi_dummy_test(:, 1:41));

% confusion matrix

results = confusionmat(table2array(bmi_dummy_test(:, 42)), preds);

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
fprintf('F1 score: %.2f%', fscore)

%% Trying the categorical verison of data with both BMI and height and weight

rng(1)

bmi_data = table([data.weight./(data.height.^2)]);
bmi_data_test = table([test.weight./(test.height.^2)]);



bmi_weight_data = [bmi_data, data];
bmi_weight_data_test = [bmi_data_test, test];


bmi_weight_data_nb = fitcnb(bmi_weight_data, 'WHO_classification',...
    'OptimizeHyperparameters', 'all',...
    'HyperparameterOptimizationOptions', struct('Optimizer', 'gridsearch', 'MaxObjectiveEvaluations', 250));


% create a 10-fold cross-validation of the model

cv = crossval(bmi_weight_data_nb, 'Kfold', 10);

% calculate cross-val accuracy

crossvalAccuracy = (1-kfoldLoss(cv))*100;

% calculate performance measures on the test set

[preds, score] = predict(bmi_weight_data_nb, bmi_weight_data_test(:, 1:17));

% confusion matrix

results = confusionmat(table2array(bmi_weight_data_test(:, 18)), preds);

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
fprintf('F1 score: %.2f%', fscore)

save bmi_weight_data_nb

%% Trying the numerically encoded verison of data with both BMI and height and weight features

rng(1)

bmi_data = table([ne.weight./(ne.height.^2)]);
bmi_data_test = table([ne_test.weight./(ne_test.height.^2)]);



bmi_weight_ne = [bmi_data, ne];
bmi_weight_ne_test = [bmi_data_test, ne_test];


bmi_weight_ne_nb = fitcnb(bmi_weight_ne, 'WHO_classification',...
    'OptimizeHyperparameters', 'all',...
    'HyperparameterOptimizationOptions', struct('Optimizer', 'gridsearch', 'MaxObjectiveEvaluations', 250));


% create a 10-fold cross-validation of the model

cv = crossval(bmi_weight_ne_nb, 'Kfold', 10);

% calculate cross-val accuracy

crossvalAccuracy = (1-kfoldLoss(cv))*100;

% calculate performance measures on the test set

[preds, score] = predict(bmi_weight_ne_nb, bmi_weight_ne_test(:, 1:17));

% confusion matrix

results = confusionmat(table2array(bmi_weight_ne_test(:, 18)), preds);

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
fprintf('F1 score: %.2f%', fscore)

save bmi_weight_ne_nb

%% Trying the dummy variable verison of data with both BMI and height and weight features

rng(1)

bmi_data = table([dummy.weight./(dummy.height.^2)]);
bmi_data_test = table([dummy_test.weight./(dummy_test.height.^2)]);



bmi_weight_dummy = [bmi_data, dummy];
bmi_weight_dummy_test = [bmi_data_test, dummy_test];


bmi_weight_dummy_nb = fitcnb(bmi_weight_dummy, 'WHO_classification',...
    'OptimizeHyperparameters', 'all',...
    'HyperparameterOptimizationOptions', struct('Optimizer', 'gridsearch', 'MaxObjectiveEvaluations', 250));


% create a 10-fold cross-validation of the model

cv = crossval(bmi_weight_dummy_nb, 'Kfold', 10);

% calculate cross-val accuracy

crossvalAccuracy = (1-kfoldLoss(cv))*100;

% calculate performance measures on the test set

[preds, score] = predict(bmi_weight_dummy_nb, bmi_weight_dummy_test(:, 1:43));

% confusion matrix

results = confusionmat(table2array(bmi_weight_dummy_test(:, 44)), preds);

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
fprintf('F1 score: %.2f%', fscore)

save bmi_weight_dummy_nb



%% Assessing feature importance using minimum reduandancy maximum relevance algorithm

[idx, scores] = fscmrmr(bmi_weight_data, 'WHO_classification');

bar(scores)
xlabel('Predictor')
ylabel('Predictor importance score')

%% just BMI

rng(1)

bmi = [bmi_data data(:, 17)];

bmi_test = [bmi_data_test test(:, 17)];

just_bmi_nb = fitcnb(bmi, 'WHO_classification',...
    'OptimizeHyperparameters', 'all',...
    'HyperparameterOptimizationOptions', struct('Optimizer', 'gridsearch', 'MaxObjectiveEvaluations', 250));


% create a 10-fold cross-validation of the model

cv = crossval(just_bmi_nb, 'Kfold', 10);

% calculate cross-val accuracy

crossvalAccuracy = (1-kfoldLoss(cv))*100;

% calculate performance measures on the test set

[preds, score] = predict(just_bmi_nb, bmi_test(:, 1));

% confusion matrix

results = confusionmat(table2array(bmi_test(:, 2)), preds);

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
fprintf('F1 score: %.2f%', fscore)

save just_bmi_nb

%% creating a nb with bmi, weight, height, and physical activity
% selection of features made through consideration of both
% predictorImportance() and fscmrmr()

rng(1)

reduced_physical_activity_train = [bmi_data, data(:, [3 4 13 17])];
    
reduced_physical_activity_test = [bmi_data_test, test(:, [3 4 13 17])];

reduced_physical_activity_nb = fitcnb(reduced_physical_activity_train, ...
    'WHO_classification',...
    'OptimizeHyperparameters', 'all',...
    'HyperparameterOptimizationOptions', struct('Optimizer', 'gridsearch', 'MaxObjectiveEvaluations', 250));


% create a 10-fold cross-validation of the model

cv = crossval(reduced_physical_activity_nb, 'Kfold', 10);

% calculate cross-val accuracy

crossvalAccuracy = (1-kfoldLoss(cv))*100;

% calculate performance measures on the test set

[preds, score] = predict(reduced_physical_activity_nb, reduced_physical_activity_test(:, 1:4));

% confusion matrix

results = confusionmat(table2array(reduced_physical_activity_test(:, 5)), preds);

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
fprintf('F1 score: %.2f%', fscore)

save reduced_physical_activity_nb

%% creating a nb with bmi, weight, height, physical activity, main transportation, 
% and number of meals daily
% selection of features made through consideration of
% predictorImportance(), fscmrmr(), and spearman rank correlation of IVs
% with the target Variable

rng(1)

semi_reduced_train = [bmi_data, data(:, [3 4 8 9 13 16 17])];
    
semi_reduced_test = [bmi_data_test, test(:, [3 4 8 9 13 16 17])];

semi_reduced_nb = fitcnb(semi_reduced_train, ...
    'WHO_classification',...
    'OptimizeHyperparameters', 'all',...
    'HyperparameterOptimizationOptions', struct('Optimizer', 'gridsearch', 'MaxObjectiveEvaluations', 250));


% create a 10-fold cross-validation of the model

cv = crossval(semi_reduced_nb, 'Kfold', 10);

% calculate cross-val accuracy

crossvalAccuracy = (1-kfoldLoss(cv))*100;

% calculate performance measures on the test set

[preds, score] = predict(semi_reduced_nb, semi_reduced_test(:, 1:7));

% confusion matrix

results = confusionmat(table2array(semi_reduced_test(:, 8)), preds);

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
fprintf('F1 score: %.2f%', fscore)

save reduced_physical_activity_nb

%% using only height and weight, no feature engineering the BMI

rng(1)

height_weight_train  = data(:, [3 4 17]);
height_weight_test = test(:, [3 4 17]);

height_weight_nb = fitcnb(height_weight_train, ...
    'WHO_classification',...
    'OptimizeHyperparameters', 'all',...
    'HyperparameterOptimizationOptions', struct('Optimizer', 'gridsearch', 'MaxObjectiveEvaluations', 250));


% create a 10-fold cross-validation of the model

cv = crossval(height_weight_nb, 'Kfold', 10);

% calculate cross-val accuracy

crossvalAccuracy = (1-kfoldLoss(cv))*100;

% calculate performance measures on the test set

[preds, score] = predict(height_weight_nb, height_weight_test(:, 1:2));

% confusion matrix

results = confusionmat(table2array(height_weight_test(:, 3)), preds);

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
fprintf('F1 score: %.2f%', fscore)

save height_weight_nb