folder='Urban100'
model='lapsrn'
scale='4'
path1="/home/mccourt/superresolution/outputs/${folder}/${model}/"
path2="/home/mccourt/results/${model}_result/${folder}/x${scale}/"

python3 -W ignore /home/mccourt/superresolution/tester_trained_ds.py \
  --model-name="${model}" \
  --image-list="/home/mccourt/SR_testing_datasets/${folder}" \
  --lr-dir="/home/mccourt/SR_testing_datasets/${folder}_x${scale}_ds" \
  --hr-dir="/home/mccourt/SR_testing_datasets/${folder}" \
  --base-dir=${path2} \
  --output-dir="/home/mccourt/superresolution/demo/outputs/${model}/${folder}" \
  --learning-rate=50000 \
  --num-epoch=500 \
  --gpu-device="cuda" \
  --log-dir="/home/mccourt/superresolution/demo/outputs/${model}/${folder}" \
  --print-every=20 \
  --attention="false" \
  --scale=$scale \
  --clip=4 \
  --lambda=.02 \
  --save="false"
