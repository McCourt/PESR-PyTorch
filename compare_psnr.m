hr_dir = 'SR_testing_datasets/Set5/';
srp_dir = 'Set5/x3/';
sr_dir = 'Set5_1/x3/sr';
clip = 10;
org_psnr = 0;
opt_psnr = 0;


hrs = dir(fullfile(hr_dir,'*.png'));
for i = 1:numel(hrs)
    hr = rgb2ycbcr(imread(fullfile(hr_dir, hrs(i).name)));
    srp = rgb2ycbcr(imread(fullfile(sr_dir, hrs(i).name)));
    sr = rgb2ycbcr(imread(fullfile(srp_dir, strrep(hrs(i).name, '.', '_LapSRN.'))));
    sr_size = size(sr);
    hr = hr(1:sr_size(1), 1:sr_size(2), :);
    hr = hr(1+clip:end-clip, 1+clip:end-clip, 1);
    sr = sr(1+clip:end-clip, 1+clip:end-clip, 1);
    srp = srp(1+clip:end-clip, 1+clip:end-clip, 1);
    org_psnr = org_psnr + psnr(sr, hr);
    opt_psnr = opt_psnr + psnr(srp, hr);
end
org_avg = org_psnr / numel(hrs);
opt_avg = opt_psnr / numel(hrs);
diff = opt_avg - org_avg;