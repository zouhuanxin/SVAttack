import os.path
import sys

sys.path.append("..")

from models.stgcn.st_gcn import STGCN_Model
from models.msg3d.msg3d import Model as MCG3D_Model
from models.hico.hi_encoder import DownstreamEncoder as HiCo_Model
from models.agcn.agcn import Model as AGCN_Model
from models.hico.hi_encoder import *
from models.hdgcn.HDGCN import Model as HDGCN_Model
from models.ctrgcn.ctrgcn import Model as CTRGCN_Model
from models.gap.ctrgcn import Model_lst_4part as GAP_Model
from models.selfgcn.SelfGCN import Model as SelfGCN_Model
from models.asgcn.as_gcn import Model as ASGCN_Model
from models.asgcn.utils.adj_learn import AdjacencyLearn as AdjacencyLearn_Model


def weights_init(model):
    with torch.no_grad():
        for child in list(model.children()):
            print("init ", child)
            for param in list(child.parameters()):
                if param.dim() == 2:
                    nn.init.xavier_uniform_(param)
    print('weights initialization finished!')


def loadSTGCN():
    load_path = '/23085412008/模型权重/STGCN/ntu60/ntu-xsub.pt'
    print(f'加载STGCN模型:{load_path}')
    graph_args = {
        "layout": "ntu-rgb+d",
        "strategy": "spatial"
    }
    stgcn = STGCN_Model(3, 60, graph_args, True)
    stgcn.eval()
    pretrained_weights = torch.load(load_path)
    stgcn.load_state_dict(pretrained_weights)
    stgcn.cuda()

    return stgcn


def loadSTGCN120():
    load_path = '/23085412008/模型权重/STGCN/ntu120/epoch90_model.pt'
    print(f'加载STGCN120模型:{load_path}')
    graph_args = {
        "layout": "ntu-rgb+d",
        "strategy": "spatial"
    }
    stgcn = STGCN_Model(3, 120, graph_args, True)
    stgcn.eval()
    pretrained_weights = torch.load(load_path)
    stgcn.load_state_dict(pretrained_weights)
    stgcn.cuda()

    return stgcn


def loadMSG3D():
    load_path = '/23085412008/模型权重/MSG3D/ntu60-xsub-joint-better.pt'
    print(f'加载MSG3D模型:{load_path}')
    mcg3d = MCG3D_Model(
        num_class=60,
        num_point=25,
        num_person=2,
        num_gcn_scales=13,
        num_g3d_scales=6,
        graph='graph.ntu_rgb_d.AdjMatrixGraph'
    )
    mcg3d.eval()
    pretrained_weights = torch.load(load_path)
    mcg3d.load_state_dict(pretrained_weights)
    mcg3d.cuda()

    return mcg3d


def loadMSG3D120():
    load_path = '/23085412008/模型权重/MSG3D/ntu120-xsub-joint.pt'
    print(f'加载MSG3D120模型:{load_path}')
    mcg3d = MCG3D_Model(
        num_class=120,
        num_point=25,
        num_person=2,
        num_gcn_scales=13,
        num_g3d_scales=6,
        graph='graph.ntu_rgb_d.AdjMatrixGraph'
    )
    mcg3d.eval()
    pretrained_weights = torch.load(load_path)
    mcg3d.load_state_dict(pretrained_weights)
    mcg3d.cuda()

    return mcg3d


def loadHiCo():
    load_path = '/23085412008/模型权重/HiCo/ntu60_xsub_joint/joint_model_best.pth.tar'
    print(f'加载HiCo模型:{load_path}')
    hico = HiCo_Model(t_input_size=150,
                      s_input_size=192,
                      kernel_size=5,
                      stride=1,
                      padding=2,
                      factor=2,
                      hidden_size=512,
                      num_head=4,
                      num_layer=1,
                      granularity=4,
                      encoder="Transformer",
                      num_class=60)
    hico.eval()
    checkpoint = torch.load(load_path)
    state_dict = checkpoint['state_dict']
    hico.load_state_dict(state_dict)
    hico.cuda()
    return hico


def loadHiCo120():
    load_path = '/23085412008/模型权重/HiCo/ntu120_xsub_joint/joint_model_best.pth.tar'
    print(f'加载HiCo120模型:{load_path}')
    hico = HiCo_Model(t_input_size=150,
                      s_input_size=192,
                      kernel_size=5,
                      stride=1,
                      padding=2,
                      factor=2,
                      hidden_size=512,
                      num_head=4,
                      num_layer=1,
                      granularity=4,
                      encoder="Transformer",
                      num_class=120)
    hico.eval()
    checkpoint = torch.load(load_path)
    state_dict = checkpoint['state_dict']
    hico.load_state_dict(state_dict)
    hico.cuda()
    return hico


def loadAGCN():
    load_path = '/23085412008/模型权重/AGCN/ntu60/ntu_cs_agcn_joint-49-31500.pt'
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

def loadAGCN120():
    load_path = '/23085412008/模型权重/AGCN/ntu120/ntu120_cs_agcn_joint-79-78720.pt'
    print(f'加载agcn120模型:{load_path}')
    agcn = AGCN_Model(
        num_class=120,
        num_point=25,
        num_person=2,
        graph='graph.ntu_rgb_d.Graph', graph_args={'labeling_mode': 'spatial'})
    agcn.eval()
    pretrained_weights = torch.load(load_path)
    agcn.load_state_dict(pretrained_weights)
    agcn.cuda()

    return agcn


def loadHDGCN():
    load_path = '/23085412008/模型权重/HDGCN/ntu-20241108T025253Z-001/ntu/cross-subject/joint_CoM_21/runs.pt'
    print(f'加载hdgcn模型:{load_path}')
    graph_args = {
        "labeling_mode": "spatial",
        "CoM": 21
    }
    hdgcn = HDGCN_Model(
        graph='graph.ntu_rgb_d_hierarchy.Graph',
        graph_args=graph_args
    )
    hdgcn.eval()
    pretrained_weights = torch.load(load_path)
    hdgcn.load_state_dict(pretrained_weights)
    hdgcn.cuda()

    return hdgcn


def loadHDGCN120():
    load_path = '/23085412008/模型权重/HDGCN/ntu120-20241108T025251Z-001/ntu120/cross-subject/joint_CoM_21/runs.pt'
    print(f'加载hdgcn120模型:{load_path}')
    graph_args = {
        "labeling_mode": "spatial",
        "CoM": 21
    }
    hdgcn = HDGCN_Model(
        num_class=120,
        graph='graph.ntu_rgb_d_hierarchy.Graph',
        graph_args=graph_args
    )
    hdgcn.eval()
    pretrained_weights = torch.load(load_path)
    hdgcn.load_state_dict(pretrained_weights)
    hdgcn.cuda()

    return hdgcn


def loadGAP():
    load_path = '/23085412008/模型权重/GAP/ntu60/ntu60-xsub.pt'
    print(f'加载gap模型:{load_path}')
    graph_args = {
        "labeling_mode": "spatial"
    }
    gap = GAP_Model(
        graph='graph.gap_ntu_rgb_d.Graph',
        graph_args=graph_args
    )
    gap.eval()
    pretrained_weights = torch.load(load_path)
    gap.load_state_dict(pretrained_weights)
    gap.cuda()

    return gap


def loadGAP120():
    load_path = '/23085412008/模型权重/GAP/ntu120/runs-110-34650.pt'
    print(f'加载gap模型:{load_path}')
    graph_args = {
        "labeling_mode": "spatial"
    }
    gap = GAP_Model(
        num_class=120,
        graph='graph.gap_ntu_rgb_d.Graph',
        graph_args=graph_args
    )
    gap.eval()
    pretrained_weights = torch.load(load_path)
    gap.load_state_dict(pretrained_weights)
    gap.cuda()

    return gap


def loadCTRGCN():
    load_path = '/23085412008/模型权重/CTRGCN/NTU60_Xsub/CTRGCN_joint_89.9/runs-60-37560.pt'
    print(f'加载ctrgcn模型:{load_path}')
    graph_args = {
        "labeling_mode": "spatial"
    }
    ctrgcn = CTRGCN_Model(
        num_class=60,
        num_point=25,
        num_person=2,
        graph='graph.ntu_rgb_d.Graph',
        graph_args=graph_args
    )
    ctrgcn.eval()
    pretrained_weights = torch.load(load_path)
    ctrgcn.load_state_dict(pretrained_weights)
    ctrgcn.cuda()

    return ctrgcn


def loadCTRGCN120():
    load_path = '/23085412008/模型权重/CTRGCN/CTRGCN_NTU120_CSub_joint_84.9-20241108T024932Z-001/CTRGCN_NTU120_CSub_joint_84.9/runs-58-57072.pt'
    print(f'加载ctrgcn模型:{load_path}')
    graph_args = {
        "labeling_mode": "spatial"
    }
    ctrgcn = CTRGCN_Model(
        num_class=120,
        num_point=25,
        num_person=2,
        graph='graph.ntu_rgb_d.Graph',
        graph_args=graph_args
    )
    ctrgcn.eval()
    pretrained_weights = torch.load(load_path)
    ctrgcn.load_state_dict(pretrained_weights)
    ctrgcn.cuda()

    return ctrgcn


def loadSelfGCN():
    load_path = '/23085412008/模型权重/SelfGCN/60X_Sub/ntu60-xsub-joint.pt'
    print(f'加载selfgcn模型:{load_path}')
    graph_args = {
        "labeling_mode": "spatial"
    }
    selfgcn = SelfGCN_Model(
        num_class=60,
        num_point=25,
        num_person=2,
        graph='graph.ntu_rgb_d.Graph',
        graph_args=graph_args
    )
    selfgcn.eval()
    pretrained_weights = torch.load(load_path)
    selfgcn.load_state_dict(pretrained_weights)
    selfgcn.cuda()

    return selfgcn


def loadSelfGCN120():
    load_path = '/23085412008/模型权重/SelfGCN/120X_Sub/ntu120-xsub-joint.pt'
    print(f'加载selfgcn120模型:{load_path}')
    graph_args = {
        "labeling_mode": "spatial"
    }
    selfgcn = SelfGCN_Model(
        num_class=120,
        num_point=25,
        num_person=2,
        graph='graph.ntu_rgb_d.Graph',
        graph_args=graph_args
    )
    selfgcn.eval()
    pretrained_weights = torch.load(load_path)
    selfgcn.load_state_dict(pretrained_weights)
    selfgcn.cuda()

    return selfgcn


def loadASGCN():
    load_path = '/23085412008/SingleViewAttack/checkpoint/epoch53_model1.pt'
    print(f'加载asgcn模型:{load_path}')
    asgcn = ASGCN_Model(
        in_channels=3,
        num_class=60,
        graph_args={'layout': 'ntu-rgb+d', 'strategy': 'spatial', 'max_hop': 4},
        edge_importance_weighting=True)
    asgcn.eval()
    pretrained_weights = torch.load(load_path)
    asgcn.load_state_dict(pretrained_weights)
    asgcn.cuda()

    return asgcn


def loadASGCN_Adj():
    load_path = '/23085412008/SingleViewAttack/checkpoint/epoch53_model2.pt'
    print(f'加载asgcn_adj模型:{load_path}')
    asgcn_adj = AdjacencyLearn_Model(150, 128, 3, 3, 128, 25)
    asgcn_adj.eval()
    pretrained_weights = torch.load(load_path)
    asgcn_adj.load_state_dict(pretrained_weights)
    asgcn_adj.cuda()

    return asgcn_adj


def getModel(AttackedModel):
    if AttackedModel == 'msg3d':
        model = loadMSG3D()
    elif AttackedModel == 'msg3d120':
        model = loadMSG3D120()
    elif AttackedModel == 'hico':
        model = loadHiCo()
    elif AttackedModel == 'hico120':
        model = loadHiCo120()
    elif AttackedModel == 'agcn':
        model = loadAGCN()
    elif AttackedModel == 'agcn120':
        model = loadAGCN120()
    elif AttackedModel == 'hdgcn':
        model = loadHDGCN()
    elif AttackedModel == 'hdgcn120':
        model = loadHDGCN120()
    elif AttackedModel == 'ctrgcn':
        model = loadCTRGCN()
    elif AttackedModel == 'ctrgcn120':
        model = loadCTRGCN120()
    elif AttackedModel == 'gap':
        model = loadGAP()
    elif AttackedModel == 'gap120':
        model = loadGAP120()
    elif AttackedModel == 'selfgcn':
        model = loadSelfGCN()
    elif AttackedModel == 'selfgcn120':
        model = loadSelfGCN120()
    elif AttackedModel == 'asgcn':
        model = loadASGCN()
    elif AttackedModel == 'asgcn_adj':
        model = loadASGCN_Adj()
    elif AttackedModel == 'stgcn120':
        model = loadSTGCN120()
    else:
        model = loadSTGCN()
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"模型总参数数量：{total_params:.2f} M")
    return model
