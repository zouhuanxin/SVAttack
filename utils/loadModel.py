import os.path
import sys
import torch

sys.path.append("..")

from models.stgcn.st_gcn import STGCN_Model
from models.agcn.agcn import Model as AGCN_Model

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

def loadAGCN():
    load_path = './checkpoint/ntu_cv_agcn_joint-49-29600.pt'
    if os.path.exists(load_path) == False:
        load_path = '../checkpoint/ntu_cv_agcn_joint-49-29600.pt'
    print(f'加载agcn模型:{load_path}')
    agcn = AGCN_Model(
        num_class=60,
        num_point=25,
        num_person=2,
        graph='graph.ntu_rgb_d.Graph', graph_args={'labeling_mode': 'spatial'})
    agcn.eval()
    pretrained_weights = torch.load(load_path)
    agcn.load_state_dict(pretrained_weights)
    agcn.cuda()

    return agcn

def getModel(AttackedModel):
    if AttackedModel == 'stgcn':
        model = loadSTGCN()
    elif AttackedModel == 'agcn':
        model = loadAGCN()
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"模型总参数数量：{total_params:.2f} M")
    return model
