import os.path
import sys
import torch

sys.path.append("..")

from models.stgcn.st_gcn import STGCN_Model

def weights_init(model):
    with torch.no_grad():
        for child in list(model.children()):
            print("init ", child)
            for param in list(child.parameters()):
                if param.dim() == 2:
                    nn.init.xavier_uniform_(param)
    print('weights initialization finished!')


def loadSTGCN():
    load_path = './checkpoint/st_gcn_model_299.pth.tar'
    if os.path.exists(load_path) == False:
        load_path = '../checkpoint/st_gcn_model_299.pth.tar'
    print(f'加载STGCN模型:{load_path}')
    graph_args = {
        "layout": "ntu-rgb+d",
        "strategy": "spatial"
    }
    stgcn = STGCN_Model(3, 60, graph_args, True)
    stgcn.eval()
    pretrained_weights = torch.load(load_path)['state_dict']
    stgcn.load_state_dict(pretrained_weights)
    stgcn.cuda()

    return stgcn


def getModel(AttackedModel):
    if AttackedModel == 'stgcn':
        model = loadSTGCN()
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"模型总参数数量：{total_params:.2f} M")
    return model
