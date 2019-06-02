#!/usr/bin/env python
import sys
import os
import os.path
import random
import numpy as np

from PIL import Image
import scipy.misc
#from skimage.measure import structural_similarity as ssim
from myssim import compare_ssim as ssim

#SCALE = 8 
SCALE = 1

def output_psnr_mse(img_orig, img_out):
    squared_error = np.square(img_orig - img_out)
    mse = np.mean(squared_error)
    psnr = 10 * np.log10(1.0 / mse)
    return psnr

def _open_img(img_p):
    F = scipy.misc.fromimage(Image.open(img_p)).astype(float)/255.0
    h, w, c = F.shape
    F = F[:h-h%SCALE, :w-w%SCALE, :]
    boundarypixels = SCALE
    F = F[boundarypixels:-boundarypixels,boundarypixels:-boundarypixels,:]
    return F

def _open_img_ssim(img_p):
    F = scipy.misc.fromimage(Image.open(img_p))#.astype(float)
    h, w, c = F.shape
    F = F[:h-h%SCALE, :w-w%SCALE, :]
    boundarypixels = SCALE 
    F = F[boundarypixels:-boundarypixels,boundarypixels:-boundarypixels,:]
    return F


def compute_psnr(ref_im, res_im):
    return output_psnr_mse(
        _open_img(os.path.join(ref_dir,ref_im)),
        _open_img(os.path.join(res_dir,res_im))
        )

def compute_mssim(ref_im, res_im):
    ref_img = _open_img_ssim(os.path.join(ref_dir,ref_im))
    res_img = _open_img_ssim(os.path.join(res_dir,res_im))
    channels = []
    for i in range(3):
        channels.append(ssim(ref_img[:,:,i],res_img[:,:,i],
        gaussian_weights=True, use_sample_covariance=False))
    return np.mean(channels)


# as per the metadata file, input and output directories are the arguments
# [_, input_dir, output_dir] = sys.argv

res_dir = os.path.join('../imgs/TrainGT/')
ref_dir = os.path.join('../imgs/TrainLR/')
#print("REF DIR")
#print(ref_dir)
input_dir = './'
output_dir = './'

runtime = -1
cpu = -1
data = -1
other = ""
readme_fnames = [p for p in os.listdir(res_dir) if p.lower().startswith('readme')]
try:
    readme_fname = readme_fnames[0]
    print("Parsing extra information from %s"%readme_fname)
    with open(os.path.join(input_dir, 'res', readme_fname)) as readme_file:
        readme = readme_file.readlines()
        lines = [l.strip() for l in readme if l.find(":")>=0]
        runtime = float(":".join(lines[0].split(":")[1:]))
        cpu = int(":".join(lines[1].split(":")[1:]))
        data = int(":".join(lines[2].split(":")[1:]))
        other = ":".join(lines[3].split(":")[1:])
except:
    print("Error occured while parsing readme.txt")
    print("Please make sure you have a line for runtime, cpu/gpu, extra data and other (4 lines in total).")
print("Parsed information:")
print("Runtime: %f"%runtime)
print("CPU/GPU: %d"%cpu)
print("Data: %d"%data)
print("Other: %s"%other)





ref_pngs = sorted([p for p in os.listdir(ref_dir) if p.lower().endswith('png')])
res_pngs = sorted([p for p in os.listdir(res_dir) if p.lower().endswith('png')])
if not (len(ref_pngs)==30 and len(res_pngs)==30):
    raise Exception('Expected 30 .png images, got %d'%len(res_pngs))




scores = []
for (ref_im, res_im) in zip(ref_pngs, res_pngs):
    print(ref_im,res_im)
    scores.append(
        compute_psnr(ref_im,res_im)
    )
    #print(scores[-1])
psnr = np.mean(scores)


scores_ssim = []
for (ref_im, res_im) in zip(ref_pngs, res_pngs):
    print(ref_im,res_im)
    scores_ssim.append(
        compute_mssim(ref_im,res_im)
        )
    #print(scores_ssim[-1])
mssim = np.mean(scores_ssim)



# the scores for the leaderboard must be in a file named "scores.txt"
# https://github.com/codalab/codalab-competitions/wiki/User_Building-a-Scoring-Program-for-a-Competition#directory-structure-for-submissions
with open(os.path.join('./', 'scores.txt'), 'w') as output_file:
    output_file.write("PSNR:%f\n"%psnr)
    output_file.write("SSIM:%f\n"%mssim)
    output_file.write("ExtraRuntime:%f\n"%runtime)
    output_file.write("ExtraPlatform:%d\n"%cpu)
    output_file.write("ExtraData:%d\n"%data)

#if __name__ == '__main__':
#
#    output_psnr_mse(_open_img(sys.argv[1]),
#                    _open_img(sys.argv[2]))





##    # unzipped submission data is always in the 'res' subdirectory
##    # https://github.com/codalab/codalab-competitions/wiki/User_Building-a-Scoring-Program-for-a-Competition#directory-structure-for-submissions
##    submission_path = os.path.join(input_dir, 'res', 'answer.txt')
##    if not os.path.exists(submission_path):
##        sys.exit('Could not find submission file {0}'.format(submission_path))
##    with open(submission_path) as submission_file:
##        submission = submission_file.read()
##    
##    # unzipped reference data is always in the 'ref' subdirectory
##    # https://github.com/codalab/codalab-competitions/wiki/User_Building-a-Scoring-Program-for-a-Competition#directory-structure-for-submissions
##    with open(os.path.join(input_dir, 'ref', 'truth.txt')) as truth_file:
##        truth = truth_file.read()
##    
#def _open_img_gray(img_p):
#    F = scipy.misc.fromimage(Image.open(img_p))
#    return F
