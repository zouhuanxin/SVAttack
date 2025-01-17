import torch

hookratio = 0.1


# 注册钩子
def register_hooks(classifier, ratio):
    hookratio = ratio
    for name, layer in classifier.named_modules():  # stgcn
        if (name == 'st_gcn_networks.4.gcn' or name == 'st_gcn_networks.5.gcn'
                or name == 'st_gcn_networks.6.gcn' or name == 'st_gcn_networks.7.gcn'
                or name == 'st_gcn_networks.8.gcn' or name == 'st_gcn_networks.9.gcn'):
            hook = layer.register_full_backward_hook(adjust_array)
    for name, layer in classifier.named_modules():  # agcn
        if ('l4.gcn1' in name or 'l5.gcn1' in name or 'l6.gcn1' in name or 'l7.gcn1' == name or 'l8.gcn1' in name
                or 'l9.gcn1' in name or 'l10.gcn1' in name):
            hook = layer.register_full_backward_hook(adjust_array)
    for name, layer in classifier.named_modules():  # ctrgcn
        if ('l4.gcn' in name or 'l5.gcn' in name or 'l6.gcn' in name or 'l7.gcn' in name or 'l8.gcn' in name or 'l9.gcn' in name or 'l10.gcn' in name):
            hook = layer.register_full_backward_hook(adjust_array)


def adjust_array(module, grad_input, grad_orutput):
    result_tensor = []
    for i in range(len(grad_input)):
        data = grad_input[i]
        data_flat = data.reshape(-1)
        size = len(data_flat)
        topk_values, topk_indices = torch.topk(data_flat, int(size * 0.9))
        data_flat[topk_indices] *= hookratio
        result_tensor.append(torch.reshape(data_flat, data.shape))
    return result_tensor
