% Second attempt using functions in MATLAB to help keep tidy
clear

load('QSAR_data.mat')
dataInitial = QSAR_data;
X1 = dataInitial(:,1:41);
Y1 = dataInitial(:,42);

%% Data cleaning/processing
columnToDuplicate = 42; % column with result which we are looking at to see if we want to duplicate or not
valueToDuplicate = 1; % we want to duplicate 1s as they're underrepresented
rowsToDuplicate = dataInitial(:,columnToDuplicate) == valueToDuplicate;
duplicatedRows = dataInitial(rowsToDuplicate,:); % all selected rows we are duplicating

UpcsaledMatrix = [dataInitial;duplicatedRows];

X = UpcsaledMatrix(:,1:41); %fomratting so can smoothly put into table form base doff code I'd written already
Y = UpcsaledMatrix(:,42);

normX = normalize(X); %improved accuracy from roughly 80% to 85% in holdout crossvalidation and %82 to 89% in k-fold
% SnormY = normalize(Y); %Decided against utilising this as it doesn't improve accuracy
normX1 = normalize(X1, 'norm'); %tried different methods of normalising but these didn't make any discernable improvement on z-score default. 'Norm' led to a significant decrease in accuracy for SVM

dataUN = [normX,Y]; %data used by KNN and Forest as it upscaling improves accuracy of these models. U - Upscaled, N - Normalised
dataN = [normX1,Y1]; % data used by SVM because it performs better without upscaling. N - Normalised

%% Formatting data for MATLAB classifiers

tblN = array2table(dataN); % converting to table form to make cross validation easier
tblN = renamevars(tblN,"dataN42", "Y"); %renaming results column to something easier

tbl = array2table(dataUN);
tbl = renamevars(tbl,"dataUN42", "Y");

%% SVM model - Hold out validation for SVM model (not upsampled)

rng('default') %----- Comes out with lower accuracy than K-fold
n = length(tblN.Y);
hpartitionN = cvpartition(n, 'HoldOut', 0.3); %Nonstratified - This is holdout as opposed to K-fold
idxTrainN = training(hpartitionN); % Selects observations in the training set for the hold-out cross-validation partition C
tblTrainHN = tblN(idxTrainN,:);
idxTestN = test(hpartitionN); % Selects observations in the test set...
tblTestHN = tblN(idxTestN, :);

inverseStep = 2; % When set too high the script takes ages to run so alter if you wish but figure when =20 is in report.
stepFactor = 2; % Costs between 0 and this number
HcvSVMtrainAccuracy = zeros(inverseStep,inverseStep); % initialising matrix to prevent MATLAB warning

for i = 1:inverseStep % for showing how cost hyperparameter changes accuracy in surf plot
    
    for j = 1:inverseStep
        HMdlSVM = fitcsvm(tblTrainHN, 'Y','BoxConstraint',1,'KernelFunction','linear', ...
        'KernelScale',1,'Standardize',false,'Solver','ISDA','ClassNames',[0,1], ...
        'Cost',[0 stepFactor*(i/inverseStep); stepFactor*(j/inverseStep) 0] );
    
        HcvMdlSVM = crossval(HMdlSVM); 
        HcvSVMtrainError = kfoldLoss(HcvMdlSVM);
        HcvSVMtrainAccuracy(i,j) = 1-HcvSVMtrainError; %Accuracy of cross validated model
    end
end
SVMxAxis = stepFactor*(1./(1:inverseStep));
SVMyAxis = stepFactor*(1./(1:inverseStep));

[maxSVMAccValue, linearIndex] = max(HcvSVMtrainAccuracy(:));
OptiHcvSVMtrainAccuracy = maxSVMAccValue;

% Convert linear index to row and column indices
[SVMAccrowIndex, SVMACCcolIndex] = ind2sub(size(HcvSVMtrainAccuracy), linearIndex);

OptiCostA = stepFactor*(SVMAccrowIndex/inverseStep);
OptiCostB = stepFactor*(SVMACCcolIndex/inverseStep); % These are the respective optimal costs
OptiHMdlSVM = fitcsvm(tblTrainHN, 'Y','BoxConstraint',1,'KernelFunction','linear', ...
        'KernelScale',1,'Standardize',false,'Solver','ISDA','ClassNames',[0,1], ...
        'Cost',[0 OptiCostA; OptiCostB 0] );
OptiHcvMdlSVM = crossval(OptiHMdlSVM);

[HcvSVMm.Pred,HcvSVMm.Scores] = kfoldPredict(OptiHcvMdlSVM);
HcvSVMm.confmat = confusionmat(OptiHcvMdlSVM.Y, HcvSVMm.Pred); %creating a confusion matrix to allow for the rest of the measures of accuracy
HcvSVMm.TP = HcvSVMm.confmat(2, 2);
HcvSVMm.TN = HcvSVMm.confmat(1, 1);
HcvSVMm.FP = HcvSVMm.confmat(1, 2);
HcvSVMm.FN = HcvSVMm.confmat(2, 1);
HcvSVMm.Accuracy = (HcvSVMm.TP + HcvSVMm.TN) / (HcvSVMm.TP ...
    + HcvSVMm.TN + HcvSVMm.FP + HcvSVMm.FN); 
HcvSVMm.sensitivity = HcvSVMm.TP / (HcvSVMm.FN + HcvSVMm.TP);
HcvSVMm.specificity = HcvSVMm.TN / (HcvSVMm.TN + HcvSVMm.FP);

%% Hold out cross validation for normalised and upsampled data

rng('default') % 'default' is for reproducability but 'shuffle' changed selection each time
n = length(tbl.Y);
hpartition = cvpartition(n, 'HoldOut', 0.3); %Nonstratified - This is holdout as opposed to K-fold
% hpartition = cvpartition(n,"Holdout",0.3,"Stratify");
idxTrain = training(hpartition); % Selects observations in the training set for the hold-out cross-validation partition C
tblTrainH = tbl(idxTrain,:);
idxTest = test(hpartition); % Selects observations in the test set...
tblTestH = tbl(idxTest, :);

%% SVM model with upsampling

OptiHMdlSVMU = fitcsvm(tblTrainH, 'Y','BoxConstraint',1,'KernelFunction','linear', ...
        'KernelScale',1,'Standardize',false,'Solver','ISDA','ClassNames',[0,1], ...
        'Cost',[0 OptiCostA; OptiCostB 0] );
OptiHcvMdlSVMU = crossval(OptiHMdlSVMU);

[HcvSVMmU.Pred,HcvSVMmU.Scores] = kfoldPredict(OptiHcvMdlSVMU);
HcvSVMmU.confmat = confusionmat(OptiHcvMdlSVMU.Y, HcvSVMmU.Pred); %creating a confusion matrix to allow for the rest of the measures of accuracy
HcvSVMmU.TP = HcvSVMmU.confmat(2, 2);
HcvSVMmU.TN = HcvSVMmU.confmat(1, 1);
HcvSVMmU.FP = HcvSVMmU.confmat(1, 2);
HcvSVMmU.FN = HcvSVMmU.confmat(2, 1);
HcvSVMmU.Accuracy = (HcvSVMmU.TP + HcvSVMmU.TN) / (HcvSVMmU.TP ...
    + HcvSVMmU.TN + HcvSVMmU.FP + HcvSVMmU.FN); 
HcvSVMmU.sensitivity = HcvSVMmU.TP / (HcvSVMmU.FN + HcvSVMmU.TP);
HcvSVMmU.specificity = HcvSVMmU.TN / (HcvSVMmU.TN + HcvSVMmU.FP);
%% KNN model - hold out - adjusting hyperparameters
HMdlKNN = fitcknn(tblTrainH, 'Y'); %When in table form, we can make a model to predict Y this way, where Y is a column

HcvMdlKNN = crossval(HMdlKNN,'KFold',10); % this is the default of this function anyway but makes it clearer

[HcvKNNm.Pred,HcvKNNm.Scores] = kfoldPredict(HcvMdlKNN);
HcvKNNm.confmat = confusionmat(HcvMdlKNN.Y, HcvKNNm.Pred); %creating a confusion matrix to allow for the rest of the measures of accuracy
HcvKNNm.TP = HcvKNNm.confmat(2, 2);
HcvKNNm.TN = HcvKNNm.confmat(1, 1);
HcvKNNm.FP = HcvKNNm.confmat(1, 2);
HcvKNNm.FN = HcvKNNm.confmat(2, 1);
HcvKNNm.Accuracy = (HcvKNNm.TP + HcvKNNm.TN) / (HcvKNNm.TP ...
    + HcvKNNm.TN + HcvKNNm.FP + HcvKNNm.FN); 
HcvKNNm.sensitivity = HcvKNNm.TP / (HcvKNNm.FN + HcvKNNm.TP);
HcvKNNm.specificity = HcvKNNm.TN / (HcvKNNm.TN + HcvKNNm.FP);

% rocKNN = rocmetrics(HcvMdlKNN.Y,HcvKNNm.Scores(:,2),1); % This plots the ROC curve for cross validated KNN model against what a random classifier would be
% plot(rocKNN)

%% Random forest an option
HForestMdl = fitcensemble(tblTrainH, 'Y','Method','Bag'); %sets default to random tree network

HcvForestMdl = crossval(HForestMdl,'KFold',10);
HcvForestTrainError = kfoldLoss(HcvForestMdl);
HcvForestTrainAccuracy = 1-HcvForestTrainError;

[HcvForestm.Pred,HcvForestm.Scores] = kfoldPredict(HcvForestMdl);
HcvForestm.confmat = confusionmat(HcvForestMdl.Y, HcvForestm.Pred); %creating a confusion matrix to allow for the rest of the measures of accuracy
HcvForestm.TP = HcvForestm.confmat(2, 2);
HcvForestm.TN = HcvForestm.confmat(1, 1);
HcvForestm.FP = HcvForestm.confmat(1, 2);
HcvForestm.FN = HcvForestm.confmat(2, 1);
HcvForestm.Accuracy = (HcvForestm.TP + HcvForestm.TN) / (HcvForestm.TP ...
    + HcvForestm.TN + HcvForestm.FP + HcvForestm.FN); 
HcvForestm.sensitivity = HcvForestm.TP / (HcvForestm.FN + HcvForestm.TP);
HcvForestm.specificity = HcvForestm.TN / (HcvForestm.TN + HcvForestm.FP);

% rocForest = rocmetrics(HcvForestMdl.Y,HcvForestm.Scores(:,2),1); % This plots the ROC curve for cross validated KNN model against what a random classifier would be
% plot(rocForest)

%% outputs - note: make SVM model not have the upscaled data

[Xs,Ys,Ts,AUCs] = perfcurve(OptiHcvMdlSVM.Y,HcvSVMm.Scores(:,2),1); %X and Y coordinates of ROC curve of the crossValidated model
[Xsu,Ysu,Tsu,AUCsu] = perfcurve(OptiHcvMdlSVMU.Y,HcvSVMmU.Scores(:,2),1); %X and Y coordinates of ROC curve of the crossValidated model
[Xf,Yf,Tf,AUCf] = perfcurve(HcvForestMdl.Y,HcvForestm.Scores(:,2),'1'); %X and Y coordinates of ROC curve of the crossValidated model
[Xk,Yk,Tk,AUCk] = perfcurve(HcvMdlKNN.Y,HcvKNNm.Scores(:,2),'1'); %X and Y coordinates of ROC curve of the crossValidated model

ColumnHeadings = ["SVM"; "SVMu"; "Forest"; "KNN"];
AUCofModels = [AUCs; AUCsu; AUCf; AUCk];
AccuracyOfModels = [HcvSVMm.Accuracy;HcvSVMmU.Accuracy;HcvForestm.Accuracy;HcvKNNm.Accuracy];
SensitivityOfModels = [HcvSVMm.sensitivity;HcvSVMmU.sensitivity;HcvForestm.sensitivity;HcvKNNm.sensitivity];
SpecificityOfModels = [HcvSVMm.specificity;HcvSVMmU.specificity;HcvForestm.specificity;HcvKNNm.specificity];

measuresTable = table(ColumnHeadings,AUCofModels, AccuracyOfModels,SensitivityOfModels,SpecificityOfModels); % table with above measures of performance can be seen in workspace

figure
plot(Xs, Ys)
hold on
plot(Xsu,Ysu)
hold on
plot(Xk, Yk)
hold on
plot(Xf, Yf)
grid on
legend({'SVM','SVMu', 'KNN', 'Forest'})
xlabel('False Positive Rate')
ylabel('True Positive Rate')

figure
surf(SVMxAxis,SVMyAxis,HcvSVMtrainAccuracy) %plots where the optimal accuracy is 
zlabel("Accuracy of SVM model")
xlabel("Cost of classifying 1 as 0")
ylabel("Cost of classifying 0 as 1")
