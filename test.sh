dataset='Set5'
model='EnhanceNet-E'
method='TTO_bicubic'
scale='4'
path1='/usr/xtmp/superresoluter/superresolution'

python3 -W ignore /usr/xtmp/superresoluter/superresolution/tester.py \
  --model-name="${model}" \
  --method-name="${method}" \
  --dataset-name="${dataset}" \
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

# folder='Urban100'
# model='lapsrn'
# scale='4'
# path1="/home/mccourt/superresolution/outputs/${folder}/${model}/"
# path2="/home/mccourt/results/${model}_result/${folder}/x${scale}/"
# 
# python3 -W ignore /home/mccourt/superresolution/tester.py \
#   --model-name="${model}" \
#   --method-name="${method}" \
#   --dataset-name="${dataset}" \
#   --image-list="/home/mccourt/SR_testing_datasets/${folder}" \
#   --lr-dir="/home/mccourt/SR_testing_datasets/${folder}_x${scale}_ds" \
#   --hr-dir="/home/mccourt/SR_testing_datasets/${folder}" \
#   --base-dir=${path2} \
#   --output-dir="/home/mccourt/superresolution/demo/outputs/${model}/${folder}" \
#   --learning-rate=50000 \
#   --num-epoch=500 \
#   --gpu-device="cuda" \
#   --log-dir="/home/mccourt/superresolution/demo/outputs/${model}/${folder}" \
#   --print-every=20 \
#   --attention="false" \
#   --scale=$scale \
#   --clip=4 \
#   --lambda=.02 \
#   --save="false"
