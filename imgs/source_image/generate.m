for myDir = ["valid/Manga109", "valid/BSDS100", "valid/DIV2K_valid", "valid/Urban100", "valid/Set5", "valid/Set14", "train/BSDS200", "train/DIV2K", "train/General100", "train/T91"]
    disp(myDir)
    myFiles = dir(fullfile(myDir, 'HR', '*.png'));
    folderName = fullfile(myDir, 'LR_matlab');
    mkdir(folderName)
    folderName = fullfile(myDir, 'HR_matlab');
    mkdir(folderName)
    for scale = [2, 4, 8]
        folderName = fullfile(myDir, 'LR_matlab', strcat('X', num2str(scale)));
        mkdir(folderName)
        folderName = fullfile(myDir, 'HR_matlab', strcat('X', num2str(scale)));
        mkdir(folderName)
        for k = 1:length(myFiles)
          baseFileName = myFiles(k).name;
          fullFileName = fullfile(myDir,'HR', baseFileName);
          hr = imread(fullFileName);
          hr = mod_crop(hr, scale);
          lr = imresize(hr, 1/scale);
          imwrite(hr, fullfile(myDir, 'HR_matlab', strcat('X', num2str(scale)), baseFileName))
          imwrite(lr, fullfile(myDir, 'LR_matlab', strcat('X', num2str(scale)), baseFileName))
        end
    end
end
