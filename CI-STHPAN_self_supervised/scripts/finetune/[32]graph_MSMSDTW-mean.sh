
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/StockForecasting" ]; then
    mkdir ./logs/StockForecasting
fi
seq_len=512
pred_len=1
model_name=Finetune_[32]graph_MSMSDTW-mean
# model_name=Finetune_[31]transfer

# model_id_name=NASDAQ
model_id_name=NYSE

data_name=stock

random_seed=2023

for k in 20
do
    for alpha in 2
    do
    python -u /home/adam/MSMS-CI-STHPAN/CI-STHPAN_self_supervised/patchtst_finetune.py \
        --random_seed $random_seed \
        --market $model_id_name \
        --is_finetune 1 \
        --is_linear_probe 0 \
        --context_points $seq_len \
        --target_points $pred_len \
        --graph 1 \
        --rel_type 8 \
        --k $k \
        --n_layers 3 \
        --n_heads 16 \
        --d_model 128 \
        --d_ff 256 \
        --dropout 0.2 \
        --head_dropout 0 \
        --n_epochs_finetune 20 \
        --alpha $alpha \
        --batch_size 1 \
        --lr 0.0001 >logs/StockForecasting/$model_name'_'$model_id_name'_'$seq_len'_k='$k'_a='$alpha.log
    done
done