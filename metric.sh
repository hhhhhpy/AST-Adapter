DATASET=${1:-"UCF"}

if [[ $DATASET == K400 ]]; then
    DATA_SET="kinetics400"
    DATA_PATH='/k400_path'
elif [[ $DATASET == UCF ]]; then
    DATA_SET="UCF101"
    DATA_PATH='/ucf_path'
elif [[ $DATASET == SSV2 ]]; then
    DATA_SET="SSV2"
    DATA_PATH='/ssv2_path'
elif [[ $DATASET == HMDB ]]; then
    DATA_SET="HMDB51"
    DATA_PATH='hmdb_path'
fi
MODEL_PATH='/uma_pretrain_model_path'

echo $DATASET
echo $DATA_PATH
echo $MODEL_PATH

torchrun="home/anaconda3/envs/ast/bin/python home/anaconda3/envs/ast/bin/torchrun"

PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512 CUBLAS_WORKSPACE_CONFIG=":4096:8" ${torchrun} --standalone --nnodes=1 \
    --nproc_per_node=1 --master_port=30953 metric.py \
    --num_frames 8 \
    --sampling_rate 2 \
    --model gb_vit_base_patch16_224 \
    --finetune ${MODEL_PATH} \
    --split "," \
    --batch_size 1 \
    --data_path ${DATA_PATH} \
    --data_set ${DATA_SET} \
    --tubelet_size 1

