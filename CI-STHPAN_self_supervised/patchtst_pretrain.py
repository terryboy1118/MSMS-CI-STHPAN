import numpy as np
import pandas as pd
import os
import torch
from torch import nn
import random
import argparse
from datetime import datetime

from src.models.patchTST import PatchTST
from src.learner import Learner
from src.callback.tracking import *
from src.callback.patch_mask import *
from src.callback.transforms import *
from src.metrics import *
from src.basics import set_device
from datautils import *

# ---------------------- #
#  Argument Parser Setup
# ---------------------- #
parser = argparse.ArgumentParser()
parser.add_argument('--random_seed', type=int, default=2023)
parser.add_argument('--dset_pretrain', type=str, default='stock')
parser.add_argument('--context_points', type=int, default=512)
parser.add_argument('--target_points', type=int, default=1)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--num_workers', type=int, default=10)
parser.add_argument('--scaler', type=str, default='standard')
parser.add_argument('--features', type=str, default='M')
parser.add_argument('--market', type=str, default='NASDAQ')
parser.add_argument('--patch_len', type=int, default=12)
parser.add_argument('--stride', type=int, default=12)
parser.add_argument('--revin', type=int, default=1)
parser.add_argument('--ci', type=int, default=1)
parser.add_argument('--graph', type=int, default=1)
parser.add_argument('--rel_type', type=int, default=0)
parser.add_argument('--k', type=int, default=20)
parser.add_argument('--n_layers', type=int, default=3)
parser.add_argument('--n_heads', type=int, default=16)
parser.add_argument('--d_model', type=int, default=128)
parser.add_argument('--d_ff', type=int, default=256)
parser.add_argument('--dropout', type=float, default=0.2)
parser.add_argument('--head_dropout', type=float, default=0)
parser.add_argument('--mask_ratio', type=float, default=0.4)
parser.add_argument('--n_epochs_pretrain', type=int, default=100)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--pretrained_model_id', type=int, default=1)
parser.add_argument('--model_type', type=str, default='based_model')
args = parser.parse_args()

print('args:', args)

# ---------------------------- #
#  Save Path Construction
# ---------------------------- #
args.save_pretrained_model = (
    'patchtst_pretrained_cw' + str(args.context_points) +
    '_patch' + str(args.patch_len) +
    '_stride' + str(args.stride) +
    '_epochs-pretrain' + str(args.n_epochs_pretrain) +
    '_mask' + str(args.mask_ratio) +
    '_revin' + str(args.revin) +
    '_ci' + str(args.ci) +
    '_graph' + str(args.graph) +
    '_rel_type' + str(args.rel_type) +
    '_k' + str(args.k)
)

# 取得目前檔案位置
current_dir = os.path.dirname(os.path.abspath(__file__))

# 完整模型儲存路徑
args.save_path = os.path.join(current_dir, 'saved_models', args.market, 'pretrained', args.save_pretrained_model)
os.makedirs(args.save_path, exist_ok=True)

# 啟動 GPU 裝置
set_device()


# ----------------------- #
#     Model 架構定義
# ----------------------- #
def get_model(c_in, args):
    num_patch = (max(args.context_points, args.patch_len) - args.patch_len) // args.stride + 1
    print('number of patches:', num_patch)

    model = PatchTST(
        c_in=c_in,
        target_dim=args.target_points,
        patch_len=args.patch_len,
        stride=args.stride,
        num_patch=num_patch,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        ci=args.ci,
        graph=args.graph,
        d_model=args.d_model,
        shared_embedding=True,
        d_ff=args.d_ff,
        dropout=args.dropout,
        head_dropout=args.head_dropout,
        act='relu',
        head_type='pretrain',
        res_attention=False
    )
    print('number of model params:', sum(p.numel() for p in model.parameters() if p.requires_grad))
    return model


# ------------------------ #
#   Learning Rate Finder
# ------------------------ #
def find_lr():
    dls = get_dls(args)
    model = get_model(dls.vars, args)
    loss_func = nn.MSELoss(reduction='mean')
    cbs = [RevInCB(dls.vars, denorm=False)] if args.revin else []
    cbs += [PatchMaskCB(args.patch_len, args.stride, args.mask_ratio)]

    learn = Learner(
        dls, model, args.graph, args.ci, args.rel_type, args.market, args.k,
        loss_func, lr=args.lr, cbs=cbs
    )
    suggested_lr = learn.lr_finder()
    print('suggested_lr:', suggested_lr)
    return suggested_lr


# ------------------------ #
#     Pretrain Function
# ------------------------ #
def pretrain_func(model_dir, lr=args.lr):
    dls = get_dls(args)
    model = get_model(dls.vars, args)
    loss_func = nn.MSELoss(reduction='mean')

    cbs = [RevInCB(dls.vars, denorm=False)] if args.revin else []
    cbs += [
        PatchMaskCB(args.patch_len, args.stride, args.mask_ratio),
        SaveModelCB(monitor='valid_loss', fname='model', path=model_dir + '/')
    ]

    learn = Learner(
        dls, model, args.graph, args.ci, args.rel_type, args.market, args.k,
        loss_func, lr=lr, cbs=cbs
    )
    learn.fit_one_cycle(n_epochs=args.n_epochs_pretrain, lr_max=lr)

    # 儲存 loss 曲線
    df = pd.DataFrame({
        'train_loss': learn.recorder['train_loss'],
        'valid_loss': learn.recorder['valid_loss']
    })
    df.to_csv(os.path.join(model_dir, 'losses.csv'), float_format='%.6f', index=False)


# ------------------------ #
#           MAIN
# ------------------------ #
if __name__ == '__main__':
    for itr in range(1):  # 多次訓練可改 range(n)
        args.pretrained_model_id = itr
        args.dset = args.dset_pretrain

        model_dir = os.path.join(args.save_path, f'model{itr}')
        os.makedirs(model_dir, exist_ok=True)

        start_time = datetime.now()
        lr = find_lr()
        pretrain_func(model_dir, lr=lr)
        end_time = datetime.now()

        print(f"itr {itr} pretraining completed！ time: {(end_time - start_time).seconds}s")
