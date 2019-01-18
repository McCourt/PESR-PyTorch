dataset='Urban100'
model='EnhanceNet-E'
method='trained_conv'
scale='4'
path="/usr/xtmp/superresoluter/superresolution"

python3 -W ignore ${path}/tester_trained_ds.py \
  --model-name="${model}" \
  --image-list="${path}/imgs/stage_one_image/${model}/${dataset}/x${scale}/" \
  --lr-dir="${path}/imgs/source_image/${dataset}/LR_PIL/x${scale}/" \
  --hr-dir="${path}/imgs/source_image/${dataset}/HR/" \
  --base-dir="${path}/imgs/stage_one_image/${model}/${dataset}/x${scale}/" \
  --output-dir="${path}/imgs/stage_two_image/${model}-${method}/${dataset}/x${scale}/" \
  --learning-rate=0.0005 \
  --num-epoch=500 \
  --gpu-device="cuda" \
  --log-dir="${path}/imgs/stage_two_image/${model}-${method}/${dataset}/x${scale}/" \
  --print-every=20 \
  --attention="false" \
  --scale=$scale \
  --clip=4 \
  --lambda=.02 \
  --save="false"
