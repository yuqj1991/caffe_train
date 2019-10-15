clc;clear all;close all;

root_dir = fullfile(getenv('HOME'), 'datasets', 'Market-1501-v15.09.15');
train_dir = fullfile(root_dir, 'bounding_box_train');% database directory
%% calculate the ID and camera for database images
train_files = dir([train_dir '/*.jpg']);
trainID = zeros(length(train_files), 1);
trainCAM = zeros(length(train_files), 1);
n_train = length(train_files);
for n = 1:n_train
    img_name = train_files(n).name;
    if strcmp(img_name(1), '-') % junk images
        trainID(n) = -1;
        trainCAM(n) = str2num(img_name(5));
    else
        trainID(n) = str2num(img_name(1:4));
        trainCAM(n) = str2num(img_name(7));
    end
end
fprintf('Train set has %4d instances\n', n_train);
save('../datamat/trainData.mat', 'trainID', 'trainCAM', 'train_files', 'train_dir');
unique_id = unique(trainID);
revers_id = zeros(max(unique_id), 1);
for n = 1:length(unique_id)
	assert (unique_id(n) >= 1);
	revers_id( unique_id(n) ) = n;
end

train_list_file = fopen('../lists/train.lst', 'w');
for n = 1:n_train
	fprintf(train_list_file, '%s/%s %d\n', train_dir, train_files(n).name, revers_id(trainID(n))-1);
	%fprintf(train_list_file, '%s/%s %4d %4d\n', train_dir, train_files(n).name, trainID(n), revers_id(trainID(n)));
end
fclose(train_list_file);
%clear *train*; close all;


%% test dir
test_dir = fullfile(root_dir, 'bounding_box_test');% database directory
%% calculate the ID and camera for database images
test_files = dir([test_dir '/*.jpg']);
testID = zeros(length(test_files), 1);
testCAM = zeros(length(test_files), 1);
n_test = length(test_files);
for n = 1:n_test
    img_name = test_files(n).name;
    if strcmp(img_name(1), '-') % junk images
        testID(n) = -1;
        testCAM(n) = str2num(img_name(5));
    else
        testID(n) = str2num(img_name(1:4));
        testCAM(n) = str2num(img_name(7));
    end
end
fprintf('Test set has %4d instances\n', n_test);
save('../datamat/testData.mat', 'testID', 'testCAM', 'test_files');
test_list_file = fopen('../lists/test.lst', 'w');
for n = 1:n_test
	fprintf(test_list_file, '%s/%s %d\n', test_dir, test_files(n).name, testID(n)-1);
end
fclose(test_list_file);
%clear test*; close all;

%% query dir
query_dir = fullfile(root_dir, 'query');% database directory
%% calculate the ID and camera for database images
query_files = dir([query_dir '/*.jpg']);
queryID = zeros(length(query_files), 1);
queryCAM = zeros(length(query_files), 1);
n_query = length(query_files);
for n = 1:n_query
    img_name = query_files(n).name;
    if strcmp(img_name(1), '-') % junk images
        queryID(n) = -1;
        queryCAM(n) = str2num(img_name(5));
    else
        queryID(n) = str2num(img_name(1:4));
        queryCAM(n) = str2num(img_name(7));
    end
end
fprintf('Query set has %4d instances\n', n_query);
save('../datamat/queryData.mat', 'queryID', 'queryCAM', 'query_files');
query_list_file = fopen('../lists/query.lst', 'w');
for n = 1:n_query
	fprintf(query_list_file, '%s/%s %d\n', query_dir, query_files(n).name, queryID(n)-1);
end
fclose(query_list_file);
