load('under_sampling.mat');

% load('matlab.mat');
% train_data = final_features_1;
% train_class = final_Class_1;


times = 10;
recall = zeros(1,times);
accuracy = zeros(1,times);
train_time = zeros(1,times);
testing_time = zeros(1,times);
true_positive_rate= zeros(1,times);

for i= 1:times
    [training_D,training_class,testing_D,testing_class] = randSample(train_data,train_class,length(train_data)/2);
    % Neural Network
    %     hiddenLayerSize = 2;
    %     net = fitnet(hiddenLayerSize);
    %     inputs = data1';
    %     targets = class1';
    %     [net,tr] = train(net,inputs,targets);
    %     testX = inputs(:,tr.testInd);
    %     testT = targets(:,tr.testInd);
    %     testY = net(testX);
    %     testing_class = testT;
    %     predict_class = testY;
    
    % Logistic Regression
%         tic;
%         bHat = glmfit(training_D,training_class,'binomial');
%         train_time(i) = toc;
%         tic
%         testing_Hat = glmval(bHat, testing_D, 'logit');
%         testing_time(i) = toc;
%         predict_class = (testing_Hat > 0.5)+0; % since this is a binomial distribution, +0 means convert logical value to double values
    
    % Decision Tree
%      tic;
%     model_decision_tree = fitctree(training_D, training_class);
%     train_time(i) = toc;
%     view(model_decision_tree, 'Mode', 'graph');
%     tic
%     [predict_class, ~, ~] = predict(model_decision_tree, testing_D);
%     testing_time(i) = toc;
    
    % Random Forest
    tic;
    numTrees = 12;
    bagger_model = TreeBagger(numTrees, training_D,training_class,'OOBPred','On');
    train_time(i) = toc;
%     view(bagger_model.Trees{1},'mode','graph');
%     view(bagger_model.Trees{2},'mode','graph');
%     view(bagger_model.Trees{3},'mode','graph');
%     view(bagger_model.Trees{4},'mode','graph');
%     view(bagger_model.Trees{5},'mode','graph');
%     view(bagger_model.Trees{6},'mode','graph');
%     view(bagger_model.Trees{7},'mode','graph');
%     view(bagger_model.Trees{8},'mode','graph');
oobErrorBaggedEnsemble = oobError(bagger_model);
figure;
plot(oobErrorBaggedEnsemble)
xlabel 'Number of grown trees';
ylabel 'Out-of-bag classification error';
    tic
    predict_class = str2double(bagger_model.predict(testing_D));
    testing_time(i) = toc;
    
    C = confusionmat(testing_class, predict_class);
%     plotroc(testing_class', predict_class');
    plotconfusion(testing_class', predict_class');
    recall(i) = C(2,2) / (C(2,1) + C(2,2));
    accuracy(i) = (C(1,1)+C(2,2))/sum(C(:));
    true_positive_rate(i) = C(2,1)/sum(C(2,:)); % rate that we missed, here is the risk, indicating how many frauds that we can't detect
end
mean_recall = mean(recall)
mean_accuracy = mean(accuracy)
mean_true_positive_rate = mean(true_positive_rate)
mean_train_time = mean(train_time)
mean_testing_time = mean(testing_time)
