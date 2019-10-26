function evaluation( model_name )

%clc;clear all;close all;
%***********************************************%
% This code runs on the Market-1501 dataset.    %
% Please modify the path to your own folder.    %
% We use the mAP and hit-1 rate as evaluation   %
%***********************************************%
% if you find this code useful in your research, please kindly cite our
% paper as,
% Liang Zheng, Liyue Sheng, Lu Tian, Shengjin Wang, Jingdong Wang, and Qi Tian,
% Scalable Person Re-identification: A Benchmark, ICCV, 2015.

% Please download Market-1501 dataset and unzip it in the "dataset" folder.

%query_feature_name = 'query.lst.fc7.mat';
%test_feature_name = 'test.lst.fc7.mat';
%model_name = 'vgg_reduce';

query_feature_name = ['query.lst.' model_name '.feature.mat'];
test_feature_name  = ['test.lst.'  model_name '.feature.mat'];

datamat_dir = fullfile(fileparts(mfilename('fullpath')), '..', 'datamat');

fprintf('data mat dir : %s\n', datamat_dir);
test_mat_path  = fullfile(datamat_dir, 'testData.mat');
query_mat_path = fullfile(datamat_dir, 'queryData.mat');
test_mat       = load(test_mat_path);
query_mat      = load(query_mat_path);

nQuery = length(query_mat.query_files);
nTest  = length(test_mat.test_files);

Hist_query = importdata(fullfile(datamat_dir, query_feature_name))';
Hist_test  = importdata(fullfile(datamat_dir, test_feature_name))';

assert(nQuery == size(Hist_query, 2));
assert(nTest == size(Hist_test, 2));
assert (all(query_mat.queryCAM >= 1)); assert (all(query_mat.queryCAM <= 6));
assert (all(test_mat.testCAM >= 1));   assert (all(test_mat.testCAM <= 6));
fprintf('Load data and features done.\n');

%% search the database and calcuate re-id accuracy
ap = zeros(nQuery, 1); % average precision
ap_pairwise = zeros(nQuery, 6); % pairwise average precision with single query (see Fig. 7 in the paper)

CMC = zeros(nQuery, nTest);
CMC_max_rerank = zeros(nQuery, nTest);

r1 = 0; % rank 1 precision with single query
r1_pairwise = zeros(nQuery, 6);% pairwise rank 1 precision with single query (see Fig. 7 in the paper)

%dist = sqdist(Hist_test, Hist_query); % distance calculate with single query. Note that Euclidean distance is equivalent to cosine distance if vectors are l2-normalized
dist = cosdist(Hist_test, Hist_query); % distance calculate with single query. For cosine distance

testID   = test_mat.testID;
queryID  = query_mat.queryID;
testCAM  = test_mat.testCAM;
queryCAM = query_mat.queryCAM;

tic;
parfor k = 1:nQuery
	fprintf('Handle (%4d / %4d)-th query. \n', k, nQuery);
    % load groud truth for each query (good and junk)
    good_index = intersect(find(testID == queryID(k)), find(testCAM ~= queryCAM(k)))';% images with the same ID but different camera from the query
    junk_index1 = find(testID == -1);% images neither good nor bad in terms of bbox quality
    junk_index2 = intersect(find(testID == queryID(k)), find(testCAM == queryCAM(k))); % images with the same ID and the same camera as the query
    junk_index = [junk_index1; junk_index2]';
    score = dist(:, k);
    
    % sort database images according Euclidean distance
    [~, index] = sort(score, 'ascend');  % single query
    
    [ap(k), CMC(k, :)] = compute_AP(good_index, junk_index, index);% compute AP for single query
    ap_pairwise(k, :) = compute_AP_multiCam(good_index, junk_index, index, queryCAM(k), testCAM); % compute pairwise AP for single query
    
    %%%%%%%%%%% calculate pairwise r1 precision %%%%%%%%%%%%%%%%%%%%
    r1_pairwise(k, :) = compute_r1_multiCam(good_index, junk_index, index, queryCAM(k), testCAM); % pairwise rank 1 precision with single query
    %%%%%%%%%%%%%% calculate r1 precision %%%%%%%%%%%%%%%%%%%%
end
FCMC = mean(CMC);
%% print result
fprintf('single query:           mAP = %f, r1 precision = %f\r cost : %.1f s\n', mean(ap), FCMC(1), toc);
%% [ap_CM, r1_CM] = draw_confusion_matrix(ap_pairwise, r1_pairwise, queryCAM);
%% fprintf('average of confusion matrix with single query:  mAP = %f, r1 precision = %f\r\n', (sum(ap_CM(:))-sum(diag(ap_CM)))/30, (sum(r1_CM(:))-sum(diag(r1_CM)))/30);

end
