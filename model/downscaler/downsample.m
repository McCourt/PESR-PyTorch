hr_rgb_input_path = '';
hr_ycbcr_output_path = '';
ds_rgb_output_path = '';
ds_ycbcr_output_path = '';

for img_name = dir(hr_rgb_input_path)
    disp(img_name)
    img = imread(fullfile(hr_input_path, img_name));
    imwrite(rgb2ycbcr(img), fullfile(hr_ycbcr_output_path, img_name));
    imwrite(resize(img, 1/4), fullfile(ds_rgb_output_path, img_name));
    imwrite(resize(rgb2ycbcr(img), 1/4), fullfile(ds_ycbcr_output_path, img_name));
end