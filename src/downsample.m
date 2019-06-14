myDir = "T91";
myFiles = dir(fullfile(myDir,'HR', '*.png'));
folderName = fullfile(myDir, 'LR_matlab');
mkdir(folderName)
for scale = [2, 4, 8]
    folderName = fullfile(myDir, 'LR_matlab', strcat('X', num2str(scale)));
    mkdir(folderName)
    for k = 1:length(myFiles)
        baseFileName = myFiles(k).name;
        fullFileName = fullfile(myDir,'HR', baseFileName);
        hr = imread(fullFileName);
        lr = imresize(hr, 1/scale);
        imwrite(lr, fullfile(myDir, 'LR_matlab', strcat('X', num2str(scale)), baseFileName))
    end
end