function [psnr_avg,ssim_avg,niqe_avg] = measure(dataset, model, scale)

hr_dir = sprintf('/usr/xtmp/superresoluter/superresolution/imgs/source_image/valid/%s/HR_matlab/X%d', dataset, scale);
sr_dir = sprintf('/usr/xtmp/superresoluter/superresolution/imgs/stage_one_image/%s/%s/X%d', dataset, model, scale);

psnr_list = 0;
ssim_list = 0;
niqe_list = 0;
dsls_list = 0;

hrs = dir(fullfile(hr_dir,'*.png'));
for i = 1:numel(hrs)
    disp(hrs(i).name)
    hr = imread(fullfile(hr_dir, hrs(i).name));
    sr = imread(fullfile(sr_dir, hrs(i).name));
    niqe_list = niqe_list + niqe(sr);
    dsls_list = dsls_list + immse(imresize(hr, 1/scale), imresize(sr, 1/scale));
    if size(hr, 3) == 3
        hr = rgb2ycbcr(hr);
    end
    if size(sr, 3) == 3
        sr = rgb2ycbcr(sr);
    end
    hr = hr(:, :, 1);
    sr = sr(:, :, 1);
    disp(NTIRE_PeakSNR_imgs(sr, hr, scale))
    psnr_list = psnr_list + NTIRE_PeakSNR_imgs(sr, hr, scale);
    ssim_list = ssim_list + NTIRE_SSIM_imgs(sr, hr, scale);
end
psnr_avg = psnr_list / numel(hrs);
ssim_avg = ssim_list / numel(hrs);
niqe_avg = niqe_list / numel(hrs);
dsls_avg = dsls_list / numel(hrs);
disp(psnr_avg)
disp(ssim_avg)
disp(niqe_avg)
disp(dsls_avg)
