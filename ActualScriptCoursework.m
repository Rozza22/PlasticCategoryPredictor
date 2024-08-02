% Second attempt using functions in MATLAB to help keep tidy
clear

load('QSAR_data.mat')
dataInitial = QSAR_data;
X1 = dataInitial(:,1:41);
Y1 = dataInitial(:,42);

%% Data cleaning
columnToDuplicate = 42;
valueToDuplicate = 1;
rowsToDuplicate = dataInitial(:,columnToDuplicate) == valueToDuplicate;
duplicatedRows = dataInitial(rowsToDuplicate,:);

UpcsaledMatrix = [dataInitial;duplicatedRows];

X = UpcsaledMatrix(:,1:41);
Y = UpcsaledMatrix(:,42);

normX = normalize(X); %improved accuracy from roughly 80% to 85% in holdout crossvalidation and %82 to 89% in k-fold
normX1 = normalize(X1);
% above comment is for when the double cv existed, latest test increased
% KNN accuracy from 79% to 85% but didn't affect  SVM

% outliersData = rmoutliers(data,"gesd"); % there aren't any outliers that should be removed, only true outliers
% X = outliersData(:,1:41);
% Y = outliersData(:,42);
% data = [X,Y]; % WIll be worth mentioning I tried it but it didnt improve
% accuracy

% finalData = outliersData;

dataUN = [normX,Y]; %data used by KNN and Forest as it upscaling improves accuracy of these models. U - Upscaled, N - Normalised
dataN = [normX1,Y1]; % data used by SVM because it performs better without upscaling. N - Normalised



%% Formatting data for MATLAB classifiers
% X = data(:,1:41);
% Y = data(:,42);

tbl = array2table(dataUN);
tbl = renamevars(tbl,"dataUN42", "Y");
%tbl.Y = tbl.data42;

%% K-Fold cross validation

% rng('default') %----- Comes out with lower accuracy than K-fold
% n = length(tbl.Y);
% K = 10; % number of folds
% i = 2; % partition number
% kpartition = cvpartition(n, 'KFold', i); %Nonstratified - This is holdout as opposed to K-fold
% idxTrain = training(kpartition,i); % Selects observations in the training set for the hold-out cross-validation partition C
% tblTrainK = tbl(idxTrain,:);
% idxNew = test(kpartition,i); % Selects observations in the test set...
% tblNewK = tbl(idxNew, :);

%% KNN model - K-fold

% KMdlKNN = fitcknn(tblTrainK, 'Y'); %When in table form, we can make a model to predict Y this way, where Y is a column
% 
% KtrainError = resubLoss(KMdlKNN);
% KNNTrainAccuracy = 1-KtrainError;

%% Mistaken attempt due to double cross validating a model

% rng('shuffle')
% n = length(tbl.Y);
% K = 10; % number of folds - will be worth seeing how changing K changes accuracy
% % i = 5; % partition number
% 
% kpartition = cvpartition(n, 'KFold', K); %Nonstratified
% 
% KNNTrainAccuracy = zeros(1,K); %initialising size of array containing accuracy for each fold
% SVMTrainAccuracy = zeros(1,K);
% 
% TotalKNNAccuracy = 0;
% TotalSVMAccuracy = 0;
% 
% for i = 1:K
%     idxTrain = training(kpartition,i); % Selects observations in the training set for the hold-out cross-validation partition
%     tblTrain = tbl(idxTrain,:);
%     idxNew = test(kpartition,i); % Selects observations in the test set...
%     tblNew = tbl(idxNew, :);
% 
%     KMdlKNN = fitcknn(tblTrain, 'Y'); %%KNN model within K-folds validation
%     % cvMdlKNN = crossval(MdlKNN); % Think this means we are cross validating within a loop which can't be right?
%     KNNTrainError = resubLoss(KMdlKNN);
%     KNNTrainAccuracy(i) = 1-KNNTrainError;
%     TotalKNNAccuracy = TotalKNNAccuracy + KNNTrainAccuracy(i);
% 
%     KMdlSVM = fitcsvm(tblTrain, 'Y'); %%SVM model within K-folds validation
%     % cvMdlSVM = crossval(MdlSVM);
%     SVMtrainError = resubLoss(KMdlSVM);
%     SVMTrainAccuracy(i) = 1-SVMtrainError;
%     TotalSVMAccuracy = TotalSVMAccuracy + SVMTrainAccuracy(i);   
% end
% 
% AvgCvKNNAccuracy = TotalKNNAccuracy/K;
% AvgCvSVMAccuracy = TotalSVMAccuracy/K;

% will also need to calculate the specificity and sensitivity


%% Hold out cross validation 

rng('default') %----- Comes out with lower accuracy than K-fold
n = length(tbl.Y);
hpartition = cvpartition(n, 'HoldOut', 0.3); %Nonstratified - This is holdout as opposed to K-fold
% hpartition = cvpartition(n,"Holdout",0.3,"Stratify");
idxTrain = training(hpartition); % Selects observations in the training set for the hold-out cross-validation partition C
tblTrainH = tbl(idxTrain,:);
idxTest = test(hpartition); % Selects observations in the test set...
tblTestH = tbl(idxTest, :);

%% SVM model - Hold out

% Weights = fscmrmr(tblTrainH,'Y');



% inverseStep = 50;
% HcvSVMtrainAccuracy = zeros(1,inverseStep);
% for i = 1:1:inverseStep

tblN = array2table(dataN);
tblN = renamevars(tblN,"dataN42", "Y");

rng('default') %----- Comes out with lower accuracy than K-fold
n = length(tblN.Y);
hpartitionN = cvpartition(n, 'HoldOut', 0.3); %Nonstratified - This is holdout as opposed to K-fold
% hpartition = cvpartition(n,"Holdout",0.3,"Stratify");
idxTrainN = training(hpartitionN); % Selects observations in the training set for the hold-out cross-validation partition C
tblTrainHN = tblN(idxTrainN,:);
idxTestN = test(hpartitionN); % Selects observations in the test set...
tblTestHN = tblN(idxTestN, :);

HMdlSVM = fitcsvm(tblTrainHN, 'Y','BoxConstraint',1,'KernelFunction','linear', ...
    'KernelScale',1,'Standardize',false,'Solver','ISDA','ClassNames',[0,1], ...
    'Cost',[0 10; 5 0] );

% --------Use for rng('default')--------
HcvMdlSVM = crossval(HMdlSVM); 
HcvSVMtrainError = kfoldLoss(HcvMdlSVM);
HcvSVMtrainAccuracy = 1-HcvSVMtrainError; %Accuracy of cross validated model

[HcvSVMm.HcvSVMPred,HcvSVMm.HcvSVMScores] = kfoldPredict(HcvMdlSVM);
HcvSVMm.HcvSVMconfmat = confusionmat(HcvMdlSVM.Y, HcvSVMm.HcvSVMPred); %creating a confusion matrix to allow for the rest of the measures of accuracy
HcvSVMm.HcvSVMTP = HcvSVMm.HcvSVMconfmat(2, 2);
HcvSVMm.HcvSVMTN = HcvSVMm.HcvSVMconfmat(1, 1);
HcvSVMm.HcvSVMFP = HcvSVMm.HcvSVMconfmat(1, 2);
HcvSVMm.HcvSVMFN = HcvSVMm.HcvSVMconfmat(2, 1);
HcvSVMm.HcvSVMAccuracy = (HcvSVMm.HcvSVMTP + HcvSVMm.HcvSVMTN) / (HcvSVMm.HcvSVMTP ...
    + HcvSVMm.HcvSVMTN  + HcvSVMm.HcvSVMFP + HcvSVMm.HcvSVMFN); % Already calculated above
HcvSVMm.HcvSVMsensitivity = HcvSVMm.HcvSVMTP / (HcvSVMm.HcvSVMFN + HcvSVMm.HcvSVMTP);
HcvSVMm.HcvSVMspecificity = HcvSVMm.HcvSVMTN / (HcvSVMm.HcvSVMTN + HcvSVMm.HcvSVMFP);
HcvSVMm.HcvSVMz = HcvSVMm.HcvSVMFP / (HcvSVMm.HcvSVMFP + HcvSVMm.HcvSVMTN);
HcvSVMm.HcvSVMX = [0;HcvSVMm.HcvSVMsensitivity;1];
HcvSVMm.HcvSVMY = [0;HcvSVMm.HcvSVMz;1];
HcvSVMm.HcvSVMAUC = trapz(HcvSVMm.HcvSVMY,HcvSVMm.HcvSVMX);  % This way is used for only binary classification

% end

% plot(1:1:inverseStep,HcvSVMtrainAccuracy)
% Linear KernelFunction is default and is the most accurate
% Kernal scale 1 is better than 'auto'
% Solver ISDA is better than default (SMO)
% HMdlSVM = fitcsvm(tblTrainH, 'Y');


% Num = 20;
% HcvSVMtrainAccuracy = zeros(1,Num);
% TotalSVMacc = 0;
% 
% for i = 1:Num
% 
%     HcvSVMtrainError = kfoldLoss(HcvMdlSVM);
%     HcvSVMtrainAccuracy(i) = 1-HcvSVMtrainError;
% 
%     TotalSVMacc = TotalSVMacc + HcvSVMtrainAccuracy(i);
% end
% 
% AvgSVMAcc = TotalSVMacc/Num;

% CompSVMcvMdl = compact(HcvMdlSVM);

% [Ypred,YCi] = predict(HcvMdlKNN,HcvMdlKNN.X);
% ActualVSprediction = [HcvMdlKNN.Y,Ypred];

%% KNN model - hold out - adjusting hyperparameters
HMdlKNN = fitcknn(tblTrainH, 'Y'); %When in table form, we can make a model to predict Y this way, where Y is a column

% trainError = resubLoss(MdlKNN);
% trainAccuracy = 1-trainError;

HcvMdlKNN = crossval(HMdlKNN,'KFold',10); % this is the default of this function anyway but makes it clearer
% HcvMdlKNN = crossval(HMdlKNN); % this is simply an improvement because it decreases overfitting
% HcvKNNtrainError = kfoldLoss(HcvMdlKNN);
% HcvKNNtrainAccuracy = 1-HcvKNNtrainError;

% trainMSE = loss(HMdlKNN,tblTrainH,"Y");
% testMSE = loss(HMdlKNN,tblTestH,"Y");

% MeasuresForHcvKNN = @(HcvMdlKNN);
% [HcvKNNtrainAccuracy, HcvKNNsensitivity, HcvKNNspecificity, HcvSVAUC] = MeasuresForHcvKNN;

[HcvKNNm.HcvKNNPred,HcvKNNm.HcvKNNScores] = kfoldPredict(HcvMdlKNN);
HcvKNNm.HcvKNNconfmat = confusionmat(HcvMdlKNN.Y, HcvKNNm.HcvKNNPred); %creating a confusion matrix to allow for the rest of the measures of accuracy
HcvKNNm.HcvKNNTP = HcvKNNm.HcvKNNconfmat(2, 2);
HcvKNNm.HcvKNNTN = HcvKNNm.HcvKNNconfmat(1, 1);
HcvKNNm.HcvKNNFP = HcvKNNm.HcvKNNconfmat(1, 2);
HcvKNNm.HcvKNNFN = HcvKNNm.HcvKNNconfmat(2, 1);
HcvKNNm.HcvKNNAccuracy = (HcvKNNm.HcvKNNTP + HcvKNNm.HcvKNNTN) / (HcvKNNm.HcvKNNTP ...
    + HcvKNNm.HcvKNNTN + HcvKNNm.HcvKNNFP + HcvKNNm.HcvKNNFN); 
HcvKNNm.HcvKNNsensitivity = HcvKNNm.HcvKNNTP / (HcvKNNm.HcvKNNFN + HcvKNNm.HcvKNNTP);
HcvKNNm.HcvKNNspecificity = HcvKNNm.HcvKNNTN / (HcvKNNm.HcvKNNTN + HcvKNNm.HcvKNNFP);
HcvKNNm.HcvKNNz = HcvKNNm.HcvKNNFP / (HcvKNNm.HcvKNNFP+HcvKNNm.HcvKNNTN);
HcvKNNm.HcvKNNX = [0;HcvKNNm.HcvKNNsensitivity;1];
HcvKNNm.HcvKNNY = [0;HcvKNNm.HcvKNNz;1];
HcvKNNm.HcvKNNAUC = trapz(HcvKNNm.HcvKNNY,HcvKNNm.HcvKNNX);  % This way is used for only binary classification

%% Random forest an option
HForestMdl = fitcensemble(tblTrainH, 'Y');

HcvForestMdl = crossval(HForestMdl,'KFold',10);
HcvForestTrainError = kfoldLoss(HcvForestMdl);
HcvForestTrainAccuracy = 1-HcvForestTrainError;

%% outputs - note: make SVM model not have the upscaled data
fprintf('Accuracy of KNN model is: %f\n', HcvKNNm.HcvKNNAccuracy);
fprintf('Accuracy of SVM model is: %f\n', HcvSVMm.HcvSVMAccuracy);
fprintf('Accuracy of RandomForest model is: %f\n', HcvForestTrainAccuracy);