%% Duplicate rows
clear
% Create a sample matrix
originalMatrix = randi([1, 10], 5, 3); % Replace with your actual matrix

% Choose the column and value for duplication condition
columnToDuplicate = 2;
valueToDuplicate = 5;

% Identify rows that meet the condition
rowsToDuplicate = originalMatrix(:, columnToDuplicate) == valueToDuplicate;

% Duplicate rows based on the condition
duplicatedRows = originalMatrix(rowsToDuplicate, :);

% Concatenate the duplicated rows to the original matrix
resultMatrix = [originalMatrix; duplicatedRows];

%% Find max value in a matrix and output position
clear
% Create a sample matrix
matrix = randi([1, 100], 3, 3, 3); % Replace with your actual matrix

% Find the maximum value and its position
[maxValue, linearIndex] = max(matrix(:));

% Convert linear index to row and column indices
[dim1Ind, dim2Ind, dim3Ind] = ind2sub(size(matrix), linearIndex);
% Display the result
disp('Original Matrix:');
disp(matrix);

disp('Maximum Value:');
disp(maxValue);

disp('Position (Row, Column):');
disp([dim1Ind, dim2Ind, dim3Ind]);

optiDim1 = 5/dim1Ind;