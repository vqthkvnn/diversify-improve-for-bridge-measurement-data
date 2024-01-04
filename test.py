import torch
from alg import alg, modelopera
from alg.algs.diversify import Diversify
from datautil.getdataloader_single import get_act_dataloader
from utils.util import set_random_seed, get_args
args = get_args()
model = torch.load("model.pth")
model.eval()
args.dataset = "cm"
args.N_WORKERS = 0
train_loader, train_loader_noshuffle, valid_loader, target_loader, _, _, _ = get_act_dataloader(
        args)

output = modelopera.accuracy(model, train_loader_noshuffle, None, "p")
print(output)