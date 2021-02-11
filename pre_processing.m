close all; clear variables; clc;

% Reading in the data, one variable for each set of cleaned data sets
% produced and output at the end

data = readtable('ObesityDataSet_raw_and_data_sinthetic.csv');
numeric_encoding_data = readtable('ObesityDataSet_raw_and_data_sinthetic.csv');
dummyvar_data = readtable('ObesityDataSet_raw_and_data_sinthetic.csv');

% Creating a copy 

data_original = readtable('ObesityDataSet_raw_and_data_sinthetic.csv');

%%
% Rename the column variables, which are mainly acronyms in Spanish, to be
% more readable

data = renamevars(data, ["Gender", "Age","Height", "Weight", "family_history_with_overweight",...
    "FAVC", 'FCVC', 'NCP', 'CAEC', 'SMOKE'...
    'CH2O', 'SCC', 'FAF', 'TUE', 'CALC', 'MTRANS', 'NObeyesdad'],...
    ["gender", "age", "height", "weight", "family_history_obesity", "fast_food_intake", 'vegetable_consumption_freq', 'number_of_meals_daily',...
    'snacking_freq', 'smoker', 'liquid_intake_daily', 'calorie_counter', ...
    'physical_activity', 'technology_usage', 'alcohol_weekly', 'main_transportation'...
    'WHO_classification']);

numeric_encoding_data = renamevars(numeric_encoding_data, ["Gender", "Age","Height", "Weight", "family_history_with_overweight",...
    "FAVC", 'FCVC', 'NCP', 'CAEC', 'SMOKE'...
    'CH2O', 'SCC', 'FAF', 'TUE', 'CALC', 'MTRANS', 'NObeyesdad'],...
    ["gender", "age", "height", "weight", "family_history_obesity", "fast_food_intake", 'vegetable_consumption_freq', 'number_of_meals_daily',...
    'snacking_freq', 'smoker', 'liquid_intake_daily', 'calorie_counter', ...
    'physical_activity', 'technology_usage', 'alcohol_weekly', 'main_transportation'...
    'WHO_classification']);

dummyvar_data = renamevars(dummyvar_data, ["Gender", "Age","Height", "Weight", "family_history_with_overweight",...
    "FAVC", 'FCVC', 'NCP', 'CAEC', 'SMOKE'...
    'CH2O', 'SCC', 'FAF', 'TUE', 'CALC', 'MTRANS', 'NObeyesdad'],...
    ["gender", "age", "height", "weight", "family_history_obesity", "fast_food_intake", 'vegetable_consumption_freq', 'number_of_meals_daily',...
    'snacking_freq', 'smoker', 'liquid_intake_daily', 'calorie_counter', ...
    'physical_activity', 'technology_usage', 'alcohol_weekly', 'main_transportation'...
    'WHO_classification']);

%% subetting the data in to the natural/raw data and synthetic

raw = data(1:498, :);
synth = data(499:end, :);

% sub-setting the modified data sets to just the raw observations as well

dummyvar_data = dummyvar_data(1:498, :);
numeric_encoding_data = numeric_encoding_data(1:498, :);

%% %% making all categorical variables categorical in type for the raw data set (not possible with the synthetic)

raw.gender = categorical(raw.gender);
raw.family_history_obesity = categorical(raw.family_history_obesity);
raw.fast_food_intake = categorical(raw.fast_food_intake);
raw.vegetable_consumption_freq = categorical(raw.vegetable_consumption_freq);
raw.number_of_meals_daily = categorical(raw.number_of_meals_daily);
raw.snacking_freq = categorical(raw.snacking_freq);
raw.smoker = categorical(raw.smoker);
raw.liquid_intake_daily = categorical(raw.liquid_intake_daily);
raw.calorie_counter = categorical(raw.calorie_counter);
raw.physical_activity = categorical(raw.physical_activity);
raw.technology_usage = categorical(raw.technology_usage);
raw.alcohol_weekly = categorical(raw.alcohol_weekly);
raw.main_transportation = categorical(raw.main_transportation);
raw.WHO_classification = categorical(raw.WHO_classification);

% And doing the same for dummyvar_data

dummyvar_data.gender = categorical(dummyvar_data.gender);
dummyvar_data.family_history_obesity = categorical(dummyvar_data.family_history_obesity);
dummyvar_data.fast_food_intake = categorical(dummyvar_data.fast_food_intake);
dummyvar_data.vegetable_consumption_freq = categorical(dummyvar_data.vegetable_consumption_freq);
dummyvar_data.number_of_meals_daily = categorical(dummyvar_data.number_of_meals_daily);
dummyvar_data.snacking_freq = categorical(dummyvar_data.snacking_freq);
dummyvar_data.smoker = categorical(dummyvar_data.smoker);
dummyvar_data.liquid_intake_daily = categorical(dummyvar_data.liquid_intake_daily);
dummyvar_data.calorie_counter = categorical(dummyvar_data.calorie_counter);
dummyvar_data.physical_activity = categorical(dummyvar_data.physical_activity);
dummyvar_data.technology_usage = categorical(dummyvar_data.technology_usage);
dummyvar_data.alcohol_weekly = categorical(dummyvar_data.alcohol_weekly);
dummyvar_data.main_transportation = categorical(dummyvar_data.main_transportation);
dummyvar_data.WHO_classification = categorical(dummyvar_data.WHO_classification);

% repeating for just the target class of the original data, in order to be
% able to modify the classes for visualisation that could be compared with
% the reference paper's classing.

data.WHO_classification = categorical(data.WHO_classification);

% repeating for the synth data set, in order to be able to modfify the
% target variable column in subsequent sections

synth.WHO_classification = categorical(synth.WHO_classification);

%% Numerically encoding the numeric_encoding data set's variables

% converting 'yes/no' columns in to logical arrays of '1/0' for 

numeric_encoding_data.gender = double(strcmp(numeric_encoding_data.gender, 'Female'));
numeric_encoding_data.family_history_obesity = double(strcmp(numeric_encoding_data.family_history_obesity, 'yes'));
numeric_encoding_data.fast_food_intake = double(strcmp(numeric_encoding_data.fast_food_intake, 'yes'));
numeric_encoding_data.smoker = double(strcmp(numeric_encoding_data.smoker, 'yes'));
numeric_encoding_data.calorie_counter = double(strcmp(numeric_encoding_data.calorie_counter, 'yes'));

% making all string variables the appropriate type for modelling and visualising

% key for snacking & alcohol:
% {1: 'no, 2: 'sometimes', 3: 'frequently', 4: 'always'}
% key for physical activity (weekly):
% {1: 'none', 2: '1-2 days', 3: '3-4 days', 4: '5-6 days'}


numeric_encoding_data.snacking_freq = double(categorical(numeric_encoding_data.snacking_freq, {'no',...
    'Sometimes', 'Frequently', 'Always'}, 'Ordinal', true));
numeric_encoding_data.alcohol_weekly = double(categorical(numeric_encoding_data.alcohol_weekly,{'no',...
    'Sometimes', 'Frequently', 'Always'}, 'Ordinal', true ));

% making the two categorical variables categorical

numeric_encoding_data.main_transportation = categorical(numeric_encoding_data.main_transportation);
numeric_encoding_data.WHO_classification = categorical(numeric_encoding_data.WHO_classification,...
    {'Insufficient_Weight','Normal_Weight','Overweight_Level_I',...
    'Overweight_Level_II', 'Obesity_Type_I','Obesity_Type_II','Obesity_Type_III'},...
    "Ordinal", true);
%% Binning target variable categories to allow comparison with reference paper 
% for all current data sets

% adding 'Overweight' category

data.WHO_classification = addcats(data.WHO_classification, 'Overweight',...
    'Before', 'Overweight_Level_I');

numeric_encoding_data.WHO_classification = addcats(numeric_encoding_data.WHO_classification, 'Overweight',...
    'Before', 'Overweight_Level_I');

dummyvar_data.WHO_classification = addcats(dummyvar_data.WHO_classification, 'Overweight',...
    'Before', 'Overweight_Level_I');

raw.WHO_classification = addcats(raw.WHO_classification, 'Overweight',...
    'Before', 'Overweight_Level_I');

synth.WHO_classification = addcats(synth.WHO_classification, 'Overweight',...
    'Before', 'Overweight_Level_I');

% Identifying both 'Overeweight_Level_I' and 'Overweight_Level_II'
% categories and changing them both to 'Overweight'

idx = find(data.WHO_classification == 'Overweight_Level_I' | ...
    data.WHO_classification == 'Overweight_Level_II');
data.WHO_classification(idx) = 'Overweight';

idx = find(numeric_encoding_data.WHO_classification == 'Overweight_Level_I' | ...
    numeric_encoding_data.WHO_classification == 'Overweight_Level_II');
numeric_encoding_data.WHO_classification(idx) = 'Overweight';

idx = find(dummyvar_data.WHO_classification == 'Overweight_Level_I' | ...
    dummyvar_data.WHO_classification == 'Overweight_Level_II');
dummyvar_data.WHO_classification(idx) = 'Overweight';

idx = find(raw.WHO_classification == 'Overweight_Level_I' | ...
    raw.WHO_classification == 'Overweight_Level_II');
raw.WHO_classification(idx) = 'Overweight';

idx = find(synth.WHO_classification == 'Overweight_Level_I' | ...
    synth.WHO_classification == 'Overweight_Level_II');
synth.WHO_classification(idx) = 'Overweight';

% Removing the two redundant categories

data.WHO_classification = removecats(data.WHO_classification,...
    {'Overweight_Level_I', 'Overweight_Level_II'});

numeric_encoding_data.WHO_classification = removecats(numeric_encoding_data.WHO_classification,...
    {'Overweight_Level_I', 'Overweight_Level_II'});

dummyvar_data.WHO_classification = removecats(dummyvar_data.WHO_classification,...
    {'Overweight_Level_I', 'Overweight_Level_II'});

raw.WHO_classification = removecats(raw.WHO_classification,...
    {'Overweight_Level_I', 'Overweight_Level_II'});

synth.WHO_classification = removecats(synth.WHO_classification,...
    {'Overweight_Level_I', 'Overweight_Level_II'});




%% visually exploring both the raw and synthetic data sets

% raw and synthetic should be compared in order to identify any
% inconsistencies in the artificial data, and thus whether it is
% appropriate for including in final results of this research

% converting synth target variable to categorical in order to plot as a
% hist to compare with other subsets and the whole data set


subplot(3, 1, 1)
hist(raw.WHO_classification)
ylabel('Raw', 'FontSize', 10, 'FontWeight','bold', 'color', 'r')
set(get(gca,'YLabel'),'Rotation',90)
a = get(gca,'XTickLabel');
set(gca,'XTickLabel',a,'fontsize',6)
subplot(3, 1, 2)
hist(synth.WHO_classification)
ylabel('Synthetic', 'FontWeight','bold', 'color', 'r', 'FontSize', 100)
set(get(gca,'YLabel'),'Rotation',90, 'VerticalAlignment','middle')
a = get(gca,'XTickLabel');
set(gca,'XTickLabel',a,'fontsize',6)
subplot(3, 1, 3)
hist(data.WHO_classification)
ylabel('All', 'FontWeight','bold', 'color', 'r', 'FontSize', 10)
set(get(gca,'YLabel'),'Rotation',90, 'VerticalAlignment','middle')
xlabel('WHO Weight Classification')
a = get(gca,'XTickLabel');
set(gca,'XTickLabel',a,'fontsize',6)

sgtitle('Distributions of Weight Groups Amongst the Dataset')

% The figure would indicate that the raw data represents the more authentic
% distribution of weight classifications in the natural population. This
% entails a degree of bias in the target variable, with 'Normal_Weight'
% being the most represented class by some margin.

% In contrast, the synthetic data produced contains more of the overweight
% and underweight instances, ostensibly to bring the overall distribution
% of the classes to be more uniform.
% 
% raw_proportions = tabulate(raw.WHO_classification);
% synthetic_proportions = tabulate(synth.WHO_classification);
% total_proportions = tabulate(data.WHO_classification);

%% Further Visualisation of data

% target_dist = gscatter(data.height, data.weight, data.WHO_classification);
% ylabel('Weight (kg)');
% xlabel('Height (m)');
% sgtitle('Classification Plotted by Height and Weight Variables');

%% Further Visualisation of data

% labels = {"Insufficient_Weight", "Normal_Weight", "Overweight_Level_I", "Overweight_Level_II","Obesity_Type_I", "Obesity_Type_II", "Obesity_Type_III"};
% plot = parallelplot(data)
% 
% plot.Jitter = 0.2;
% plot.GroupVariable = 'WHO_classification';

%% Preliminary Principal Component Analysis (PCA) with numerically encoded data

% numdata = table2array(numeric_encoding_data(:, 1:end-2));
% 
% [pcs, scrs, ~, ~, pexp] = pca(numdata);
% 
% pareto(pexp)
% 
% scatter(scrs(:,1), scrs(:,2), 10, data.WHO_classification)
% xlabel('Principal Component 1 (approx. 94% variance)')
% ylabel('Principal Component 2 (approx. 6% variance)')
% sgtitle('PCA Analysis')
% c = colorbar('Ticks', [1:6], 'TickLabels',{"Insufficient_Weight", "Normal_Weight", "Overweight","Obesity_Type_I", "Obesity_Type_II", "Obesity_Type_III"});


%% creating dummy variables for the dummyvar_data

% Making the 'physical_activity' column suitable for the dummvar() function
% by adding one to each value to get rid of zeros

dummyvar_data.physical_activity = addcats(dummyvar_data.physical_activity, '4',...
    'After', '3');

idx = find(dummyvar_data.physical_activity == '3');
dummyvar_data.physical_activity(idx) = '4';

idx = find(dummyvar_data.physical_activity == '2');
dummyvar_data.physical_activity(idx) = '3';

idx = find(dummyvar_data.physical_activity == '1');
dummyvar_data.physical_activity(idx) = '2';

idx = find(dummyvar_data.physical_activity == '0');
dummyvar_data.physical_activity(idx) = '1';

% removing the zero category

dummyvar_data.physical_activity = removecats(dummyvar_data.physical_activity,...
    {'0'});

%% isolating the categorical variables to convert to dummy variables and
% then append together with the numeric variables and the target variable



categories = dummyvar_data{:, [1 5 6 7 8 9 10 11 12 13 14 15 16]};
numerics = dummyvar_data(:, 2:4);


dummys_gender = table(dummyvar(dummyvar_data.gender), 'VariableNames', {'dummys_gender'});
dummys_family_history_obesity = table(dummyvar(dummyvar_data.family_history_obesity),...
    'VariableNames', {'dummys_family_history_obesity'});
dummys_fast_food_intake = table(dummyvar(dummyvar_data.fast_food_intake),...
    'VariableNames', {'dummys_fast_food_intake'});
dummys_vegetable_consumption_freq = table(dummyvar(dummyvar_data.vegetable_consumption_freq), ...
    'VariableNames', {'dummys_vegetable_consumption_freq'});
dummys_number_of_meals_daily = table(dummyvar(dummyvar_data.number_of_meals_daily), ...
    'VariableNames', {'dummys_number_of_meals_daily'});
dummys_snacking_freq = table(dummyvar(dummyvar_data.snacking_freq), ...
    'VariableNames', {'dummys_snacking_freq'});
dummys_smoker = table(dummyvar(dummyvar_data.smoker), ...
    'VariableNames', {'dummys_smoker'});
dummys_liquid_intake_daily = table(dummyvar(dummyvar_data.liquid_intake_daily), ...
    'VariableNames', {'dummys_liquid_intake_daily'});
dummys_calorie_counter = table(dummyvar(dummyvar_data.calorie_counter), ...
    'VariableNames', {'dummys_calorie_counter'});
dummys_physical_activity = table(dummyvar(dummyvar_data.physical_activity), ...
    'VariableNames', {'dummys_physical_activity'});
dummys_dummys_technology_usage = table(dummyvar(dummyvar_data.technology_usage), ...
    'VariableNames', {'dummys_technology_usage'});
dummys_alcohol_weekly = table(dummyvar(dummyvar_data.alcohol_weekly), ...
    'VariableNames', {'dummys_alcohol_weekly'});
dummys_main_transportation = table(dummyvar(dummyvar_data.main_transportation), ...
    'VariableNames', {'dummys_main_transportation'});
%%
% concatenating together all the dummy variables and the numeric variables 
% and target variable

% making the target variable a table in order to be able to concatenate it
% with the rest of the variables

dummys_WHO_classification = table(dummyvar_data.WHO_classification, 'VariableNames',...
    {'WHO_classification'});

dummyvar_data = [numerics, dummys_gender, dummys_family_history_obesity,...
    dummys_fast_food_intake, dummys_vegetable_consumption_freq, ...
    dummys_number_of_meals_daily, dummys_snacking_freq, dummys_smoker, ...
    dummys_liquid_intake_daily, dummys_calorie_counter, dummys_physical_activity, ...
    dummys_dummys_technology_usage, dummys_alcohol_weekly, dummys_main_transportation, ...
    dummys_WHO_classification]


%% spliting the data sets in to test and training sets

% Setting the seed of the random number generator to ensure reproducibility
% of results and fair comparabilit for train/test split between different
% data sets

rng(5)

% splitting standard categorical data

indices = randperm(size(data, 1));

% split into 80% : 20% train/test split

x = round(size(data, 1)*0.8);

train = data(indices(1:x), :);
test = data(indices(x+1:end), :);

Xtrain = data(indices(1:x),1:end-1);
Xtest = data(indices(x+1:end),1:end-1);
ytrain = data(indices(1:x),end);
ytest = data(indices(x+1:end),end);

%% Doing the same for just the raw portion of the categorical data

% Setting the seed of the random number generator to ensure reproducibility
% of results and fair comparabilit for train/test split between different
% data sets

rng(5)

indices = randperm(size(raw, 1));

% split into 0.8/0.2 train/test split

x = round(size(raw, 1)*0.8);

raw_train = raw(indices(1:x), :);
raw_test = raw(indices(x+1:end), :);

rawXtrain = raw(indices(1:x),1:end-1);
rawXtest = raw(indices(x+1:end),1:end-1);
rawytrain = raw(indices(1:x),end);
rawytest = raw(indices(x+1:end),end);

%% Repeating for the numerically-encoded data

% Setting the seed of the random number generator to ensure reproducibility
% of results and fair comparabilit for train/test split between different
% data sets

rng(5)

% subsetting just the raw data

numeric_encoding_data = numeric_encoding_data(1:498, :);

indices = randperm(size(numeric_encoding_data, 1));

% split into 0.8/0.2 train/test split

x = round(size(numeric_encoding_data, 1)*0.8);

numeric_encoding_train = numeric_encoding_data(indices(1:x), :);
numeric_encoding_test = numeric_encoding_data(indices(x+1:end), :);

numeric_encoding_Xtrain = numeric_encoding_data(indices(1:x),1:end-1);
numeric_encoding_Xtest = numeric_encoding_data(indices(x+1:end),1:end-1);
numeric_encoding_ytrain = numeric_encoding_data(indices(1:x),end);
numeric_encoding_ytest = numeric_encoding_data(indices(x+1:end),end);

%% Repeating for the dummy variable data

% Setting the seed of the random number generator to ensure reproducibility
% of results and fair comparabilit for train/test split between different
% data sets

rng(5)

indices = randperm(size(dummyvar_data, 1));

% split into 0.8/0.2 train/test split

x = round(size(dummyvar_data, 1)*0.8);

dummy_train = dummyvar_data(indices(1:x), :);
dummy_test = dummyvar_data(indices(x+1:end), :);

dummy_Xtrain = dummyvar_data(indices(1:x),1:end-1);
dummy_Xtest = dummyvar_data(indices(x+1:end),1:end-1);
dummy_ytrain = dummyvar_data(indices(1:x),end);
dummy_ytest = dummyvar_data(indices(x+1:end),end);


%% Writing pre-processed data to .csv files

% % whole data set cleaned, for attempting modelling with the synthetic data
% % included
% 
% writetable(data, 'prepped_data.csv');
% 
% % Just the raw data
% 
% writetable(raw, 'prepped_raw_data.csv');
% writetable(raw_train, 'prepped_raw_training_data.csv');
% writetable(raw_test, 'prepped_raw_test_data.csv');
% 
% % the numeric encoding data
% 
% writetable(numeric_encoding_train, 'numeric_encoding_train.csv');
% writetable(numeric_encoding_test, 'numeric_encoding_test.csv');
% 
% % and finally, the dummy variable data
% 
% writetable(dummy_train, 'dummy_train.csv');
% writetable(dummy_test, 'dummy_test.csv');
% 
% 
% 
% 
% 




    