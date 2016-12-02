function [train_input, train_target, valid_input, valid_target, test_input, test_target, vocab] = load_data(N)
% This method loads the training, validation and test set.
% It also divides the training set into mini-batches.
% Inputs:
%   N: Mini-batch size.
% Outputs:
%   train_input: An array of size D X N X M, where
%                 D: number of input dimensions (in this case, 3).
%                 N: size of each mini-batch (in this case, 100).
%                 M: number of minibatches.
%   train_target: An array of size 1 X N X M.
%   valid_input: An array of size D X number of points in the validation set.
%   test: An array of size D X number of points in the test set.
%   vocab: Vocabulary containing index to word mapping.

load data.mat;
numdims = size(data.trainData, 1);   % 4 row
D = numdims - 1;                     % 1-3 is inout ,4 is output
M = floor(size(data.trainData, 2) / N);  % N batchsize size,M is the batch number.
train_input = reshape(data.trainData(1:D, 1:N * M), D, N, M);
train_target = reshape(data.trainData(D + 1, 1:N * M), 1, N, M);  %targrt is the groud truth.
valid_input = data.validData(1:D, :);
valid_target = data.validData(D + 1, :);
test_input = data.testData(1:D, :);
test_target = data.testData(D + 1, :);
vocab = data.vocab;             % vocabulary set .we have 250 different words.
end
