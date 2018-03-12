%load input
InputDirectoryPath = uigetdir('select file path');
X_train = importdata(strcat(InputDirectoryPath,'\X_train.mat'));
y_train = importdata(strcat(InputDirectoryPath,'\y_train.mat'));
X_test = importdata(strcat(InputDirectoryPath,'\X_test.mat'));
y_test = importdata(strcat(InputDirectoryPath,'\y_test.mat'));
y_test = y_test';
y_train = y_train';
k = 7;
%Train and predict classes for test data
Mdl = fitcknn(X_train,y_train,'NumNeighbors',k,'NSMethod','exhaustive','Standardize',1);
label_KNN = predict(Mdl,X_test);

%Find accuracy by comparing already available classes of test data
cnt =0;
for i =1 :size(label_KNN)
    if(label_KNN(i) == y_test(i))
        cnt = cnt+1;
    end
end
fprintf('KNN Accuracy = %2.2f%%\n',cnt*100/i);


classes = unique(y_train);  % unique class lebels
label_SVM = zeros(size(X_test,1),size(classes,1));
 for j = 1:numel(classes);
     indx = eq(y_train,classes(j)); % Create binary classes for each classifier
     SVMModel = fitcsvm(X_train,indx,'ClassNames',[false true],'KernelFunction','polynomial','PolynomialOrder',2);
     [label_SVM(:,j),scores] = predict(SVMModel,X_test); 
 end
count =0;
%compute accuracy
  for i = 1 : size(label_SVM(:,1))
      % if k'th SVM gives positive and k = class in ground truth then its a
      % match!!
      if(find(label_SVM(i,:)) == y_test(i))
          count = count+1;
      end
  end
  fprintf('SVM Accuracy = %2.2f%%\n',(count*100/i));


y_train = full(ind2vec(y_train));
%create and train the network
net= patternnet(25);
net = train(net,X_train',y_train);
%test network on test data
label_ANN = net(X_test');
label_ANN= vec2ind(label_ANN);
%calculate accuracy
cnt =0;
for i =1:length(label_ANN)
    if label_ANN(i)== y_test(i)
        cnt = cnt +1;
    end
end
fprintf('ANN Accuracy = %2.2f%%\n',cnt*100/i);
%compute accuracy of ensemble
cnt=0;
label_ANN = label_ANN';
for i =1:length(label_ANN)
    temp = [label_KNN(i),label_ANN(i),find(label_SVM(i,:))];
    if mode(temp)== y_test(i)
        cnt = cnt +1;
    end
end
fprintf('Ensemble Accuracy = %2.2f%%\n',cnt*100/i);
