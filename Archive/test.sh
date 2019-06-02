dataset='Set5'
model='EnhanceNet-E'
method='TTO_1hermite'
scale='4'
path1='/usr/xtmp/superresoluter/superresolution'

python3 -W ignore /usr/xtmp/superresoluter/superresolution/tester_withhermite.py \
  --model-name="${model}" \
  --method-name="${method}" \
  --image-list="${path1}/imgs/stage_one_image/${model}/${dataset}/x${scale}/" \
  --lr-dir="${path1}/imgs/source_image/${dataset}/LR_PIL/x${scale}/" \
  --hr-dir="${path1}/imgs/source_image/${dataset}/HR/" \
  --stage1-dir="${path1}/imgs/stage_one_image/${model}/${dataset}/x${scale}/" \
  --output-dir="${path1}/imgs/stage_two_image/${model}-${method}/${dataset}/x${scale}/" \
  --learning-rate=50000 \
  --num-epoch=500 \
  --gpu-device="cuda" \
  --log-dir="${path1}/imgs/stage_two_image/${model}-${method}/${dataset}/x${scale}/" \
  --print-every=20 \
  --attention="false" \
  --scale=$scale \
  --clip=4 \
  --lambda=.02 \
  --save="true"
