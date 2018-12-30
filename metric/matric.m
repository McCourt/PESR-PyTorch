hr_path = '';
sr_path = '';
srp_path = '';
lr_path = '';

sr_psnrs = zeros(length(dir(hr_path)));
srp_psnrs = zeros(length(dir(hr_path)));
cnt = 1;
for img_name = dir(hr_path)
    disp(img_name)
    hr_img = imread(fullfile(hr_path, img_name));
    hr_y = hr_img(:, :, 1);
    sr_img = imread(fullfile(sr_path, img_name));
    sr_y = sr_img(:, :, 1);
    srp_img = imread(fullfile(srp_path, img_name));
    srp_y = srp_img(:, :, 1);
    lr_img = imread(fullfile(lr_path, img_name));
    lr_y = lr_img(:, :, 1);
    
    sr_psnrs(cnt) = psnr(hr_img, sr_img);
    srp_psnrs(cnt) = psnr(hr_img, srp_img);
    cnt = cnt + 1;
end

disp(mean(sr_psnrs))
disp(mean(srp_psnrs))