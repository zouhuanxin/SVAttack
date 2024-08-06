import torch


def register_hooks(classifier):
    for name, layer in classifier.named_modules():
        if name == 'st_gcn_networks.0.gcn' or name == 'st_gcn_networks.0.gcn.conv' \
                or name == 'st_gcn_networks.1.gcn' or name == 'st_gcn_networks.1.gcn.conv' \
                or name == 'st_gcn_networks.2.gcn' or name == 'st_gcn_networks.2.gcn.conv' \
                or name == 'st_gcn_networks.3.gcn' or name == 'st_gcn_networks.3.gcn.conv' \
                or name == 'st_gcn_networks.4.gcn' or name == 'st_gcn_networks.4.gcn.conv' \
                or name == 'st_gcn_networks.5.gcn' or name == 'st_gcn_networks.5.gcn.conv' \
                or name == 'st_gcn_networks.6.gcn' or name == 'st_gcn_networks.6.gcn.conv' \
                or name == 'st_gcn_networks.7.gcn' or name == 'st_gcn_networks.7.gcn.conv' \
                or name == 'st_gcn_networks.8.gcn' or name == 'st_gcn_networks.8.gcn.conv' \
                or name == 'st_gcn_networks.9.gcn' or name == 'st_gcn_networks.9.gcn.conv':
            hook = layer.register_full_backward_hook(adjust_array)
    for name, layer in classifier.named_modules():
        if ('l1.gcn1' in name or 'l2.gcn1' in name or 'l3.gcn1' in name or 'l4.gcn1' in name
                or 'l5.gcn1' in name or 'l6.gcn1' in name or 'l7.gcn1' == name or 'l8.gcn1' in name
                or 'l9.gcn1' in name or 'l10.gcn1' in name):
            hook = layer.register_full_backward_hook(adjust_array)
    for name, layer in classifier.named_modules():
        if ('gcn3d1' in name or 'sgcn1' in name or 'gcn3d2' in name or 'sgcn2' in name
                or 'gcn3d3' in name or 'sgcn3' in name):
            hook = layer.register_full_backward_hook(adjust_array)


def remove_hooks(self):
    for hook in self.hooks:
        hook.remove()
    self.hooks = []


def scale_gradient(module, grad_input, grad_output):
    for i in range(len(grad_input)):
        grad_input[i].mul_(0.3)
    return grad_input


def adjust_array(module, grad_input, grad_output):
    result_tensor = []
    for i in range(len(grad_input)):
        data = grad_input[i]
        data_flat = data.reshape(-1)
        size = len(data_flat)
        topk_values, topk_indices = torch.topk(data_flat, int(size/4))
        data_flat[topk_indices] *= 0.5
        result_tensor.append(torch.reshape(data_flat, data.shape))
    return result_tensor
