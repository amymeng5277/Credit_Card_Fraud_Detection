function [ data,class,data_test,test_class ] = randSample(data_a, data_class, num)
index_random = randperm(length(data_a));
data = data_a(index_random(1:num),:);
class = data_class(index_random(1:num),:);
data_a(index_random(1:num),:) = [];
data_class(index_random(1:num),:) = [];
data_test = data_a;
test_class = data_class;