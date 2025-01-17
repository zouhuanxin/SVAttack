import numpy as np
import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch.nn.functional as F

class DepthWiseConv(nn.Module):
    def __init__(self, in_ch, kn_sz, sd, pd) -> None:
        super().__init__()
        self.dw_conv = nn.Conv1d(in_channels=in_ch, out_channels=in_ch, kernel_size=kn_sz, stride=sd, padding=pd,
                                 groups=in_ch)

    def forward(self, input_tensor):
        return self.dw_conv(input_tensor.transpose(2, 1)).transpose(2, 1)


class UDM(nn.Module):
    """Unified Donwsampling Module"""

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, factor) -> None:
        super().__init__()
        self.conv = DepthWiseConv(in_channels, kernel_size, stride, padding)
        self.norm = nn.LayerNorm(out_channels)
        self.relu = nn.ReLU(True)
        self.maxpooling = nn.MaxPool1d(factor)

    def forward(self, input_tensor):
        src = self.conv(input_tensor)
        src = self.norm(src)
        src = self.relu(src)
        src = self.maxpooling(src.transpose(2, 1)).transpose(2, 1)
        return src


class HiEncoder(nn.Module):
    """Two branch hierarchical encoder with multi-granularity"""

    def __init__(self, t_input_size, s_input_size,
                 kernel_size, stride, padding, factor,
                 hidden_size, num_head, num_layer,
                 granularity,
                 encoder,
                 ) -> None:
        super().__init__()
        self.d_model = hidden_size
        self.granularity = granularity
        self.encoder = encoder

        # temproal and spatial branch embedding layers
        self.t_embedding = nn.Sequential(
            nn.Linear(t_input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(True),
            nn.Linear(hidden_size, hidden_size),
        )
        self.s_embedding = nn.Sequential(
            nn.Linear(s_input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(True),
            nn.Linear(hidden_size, hidden_size),
        )

        # downsampling modules
        self.t_downsample = UDM(hidden_size, hidden_size, kernel_size, stride, padding, factor)
        self.s_downsample = UDM(hidden_size, hidden_size, kernel_size, stride, padding, factor)

        # seq2seq encoders
        if encoder == "GRU":
            self.t_encoder = nn.GRU(input_size=self.d_model, hidden_size=self.d_model // 2, num_layers=num_layer,
                                    batch_first=True, bidirectional=True)
            self.s_encoder = nn.GRU(input_size=self.d_model, hidden_size=self.d_model // 2, num_layers=num_layer,
                                    batch_first=True, bidirectional=True)
        elif encoder == "LSTM":
            self.t_encoder = nn.LSTM(input_size=self.d_model, hidden_size=self.d_model // 2, num_layers=num_layer,
                                     batch_first=True, bidirectional=True)
            self.s_encoder = nn.LSTM(input_size=self.d_model, hidden_size=self.d_model // 2, num_layers=num_layer,
                                     batch_first=True, bidirectional=True)
        elif encoder == "Transformer":
            encoder_layer = TransformerEncoderLayer(self.d_model, num_head, self.d_model, batch_first=True)
            self.t_encoder = TransformerEncoder(encoder_layer, num_layer)
            self.s_encoder = TransformerEncoder(encoder_layer, num_layer)
        else:
            raise ValueError("Unknown encoder!")

    def forward(self, xc, xp):
        # Given the time-majored domain input sequence xc
        # and the space-majored domain input sequence xp

        if self.encoder == "GRU" or self.encoder == "LSTM":
            self.t_encoder.flatten_parameters()
            self.s_encoder.flatten_parameters()

        # embedding
        xc = self.t_embedding(xc)  # temporal domain
        xp = self.s_embedding(xp)  # spatial domain

        # represents skeleton sequences into multiple feature of
        # different granularities from both temporal and spatial domains
        if self.encoder == "GRU" or self.encoder == "LSTM":
            vc, _ = self.t_encoder(xc)
            vp, _ = self.s_encoder(xp)
        else:
            vc = self.t_encoder(xc)
            vp = self.s_encoder(xp)

        # implementation using amax for the TMP runs faster than using MaxPool1D
        # not support pytorch < 1.7.0
        vc = vc.amax(dim=1).unsqueeze(1)
        vp = vp.amax(dim=1).unsqueeze(1)

        for i in range(1, self.granularity):
            xc = self.t_downsample(xc)  # obtain clips of different temporal granularities
            xp = self.s_downsample(xp)  # obtain parts of different spatial granularities

            if self.encoder == "GRU" or self.encoder == "LSTM":
                vc_i, _ = self.t_encoder(xc)
                vp_i, _ = self.s_encoder(xp)
            else:
                vc_i = self.t_encoder(xc)
                vp_i = self.s_encoder(xp)

            vc_i = vc_i.amax(dim=1).unsqueeze(1)
            vp_i = vp_i.amax(dim=1).unsqueeze(1)

            vc = torch.cat([vc, vc_i], dim=1)
            vp = torch.cat([vp, vp_i], dim=1)

        return vc, vp


class PretrainingEncoder(nn.Module):
    """hierarchical encoder network + projectors"""

    def __init__(self, t_input_size, s_input_size,
                 kernel_size, stride, padding, factor,
                 hidden_size, num_head, num_layer,
                 granularity,
                 encoder,
                 num_class=60,
                 ):
        super(PretrainingEncoder, self).__init__()

        self.d_model = hidden_size

        self.hi_encoder = HiEncoder(
            t_input_size, s_input_size,
            kernel_size, stride, padding, factor,
            hidden_size, num_head, num_layer,
            granularity,
            encoder,
        )

        # clip level feature projector
        self.clip_proj = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.ReLU(True),
            nn.Linear(self.d_model, num_class)
        )

        # part level feature projector
        self.part_proj = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.ReLU(True),
            nn.Linear(self.d_model, num_class)
        )

        # temporal domain level feature projector
        self.td_proj = nn.Sequential(
            nn.Linear(granularity * self.d_model, self.d_model),
            nn.ReLU(True),
            nn.Linear(self.d_model, num_class)
        )

        # spatial domain level feature projector
        self.sd_proj = nn.Sequential(
            nn.Linear(granularity * self.d_model, self.d_model),
            nn.ReLU(True),
            nn.Linear(self.d_model, num_class)
        )

        # instance level feature projector
        self.instance_proj = nn.Sequential(
            nn.Linear(2 * granularity * self.d_model, self.d_model),
            nn.ReLU(True),
            nn.Linear(self.d_model, num_class)
        )

    def forward(self, xc, xp):
        # we use concatenation as our feature fusion method

        # obtain clip and part level representations
        vc, vp = self.hi_encoder(xc, xp)

        # concatenate different granularity features as temproal and spatial domain representations
        vt = vc.reshape(vc.shape[0], -1)
        vs = vp.reshape(vp.shape[0], -1)

        # same for instance level representation
        vi = torch.cat([vt, vs], dim=1)

        # projection
        zc = self.clip_proj(vc)
        zp = self.part_proj(vp)

        zt = self.td_proj(vt)
        zs = self.sd_proj(vs)

        zi = self.instance_proj(vi)

        return zc, zp, zt, zs, zi


class DownstreamEncoder(nn.Module):
    """hierarchical encoder network + classifier"""

    def __init__(self, t_input_size, s_input_size,
                 kernel_size, stride, padding, factor,
                 hidden_size, num_head, num_layer,
                 granularity,
                 encoder,
                 num_class=60,
                 ):
        super(DownstreamEncoder, self).__init__()

        self.d_model = hidden_size

        self.hi_encoder = HiEncoder(
            t_input_size, s_input_size,
            kernel_size, stride, padding, factor,
            hidden_size, num_head, num_layer,
            granularity,
            encoder,
        )

        # linear classifier
        self.fc = nn.Linear(2 * granularity * self.d_model, num_class)

    def forward(self, data_numpy, frame, knn_eval=False):
        data_numpy = data_numpy.reshape(data_numpy.shape[1], data_numpy.shape[2], data_numpy.shape[3],
                                        data_numpy.shape[4])
        data_numpy = data_numpy.cpu().numpy()
        data_numpy = self.crop_subsequence(data_numpy, frame)
        data_numpy = data_numpy.reshape(1, data_numpy.shape[0], data_numpy.shape[1], data_numpy.shape[2], data_numpy.shape[3])
        data_numpy = torch.tensor(data_numpy).cuda()
        N, C, T, V, M = data_numpy.shape
        qc_joint = data_numpy.permute(0, 2, 4, 3, 1)
        qc_joint = qc_joint.reshape(N, T, M * V * C).float()
        qp_joint = data_numpy.permute(0, 4, 3, 2, 1)
        qp_joint = qp_joint.reshape(N, M * V, T * C).float()

        vc, vp = self.hi_encoder(qc_joint, qp_joint)

        vt = vc.reshape(vc.shape[0], -1)
        vs = vp.reshape(vp.shape[0], -1)

        vi = torch.cat([vt, vs], dim=1)

        if knn_eval:  # return last layer features during  KNN evaluation (action retrieval)
            return vi
        else:
            return self.fc(vi)

    def crop_subsequence(self, input_data, num_of_frames, l_ratio=[0.95], output_size=64):

        C, T, V, M = input_data.shape

        if l_ratio[0] == 0.5:
            # if training , sample a random crop

            min_crop_length = 64
            scale = np.random.rand(1) * (l_ratio[1] - l_ratio[0]) + l_ratio[0]
            temporal_crop_length = np.minimum(np.maximum(int(np.floor(num_of_frames * scale)), min_crop_length),
                                              num_of_frames)

            start = np.random.randint(0, num_of_frames - temporal_crop_length + 1)
            temporal_crop = input_data[:, start:start + temporal_crop_length, :, :]

            temporal_crop = torch.tensor(temporal_crop, dtype=torch.float)
            temporal_crop = temporal_crop.permute(0, 2, 3, 1).contiguous().view(C * V * M, temporal_crop_length)
            temporal_crop = temporal_crop[None, :, :, None]
            temporal_crop = F.interpolate(temporal_crop, size=(output_size, 1), mode='bilinear', align_corners=False)
            temporal_crop = temporal_crop.squeeze(dim=3).squeeze(dim=0)
            temporal_crop = temporal_crop.contiguous().view(C, V, M, output_size).permute(0, 3, 1,
                                                                                          2).contiguous().numpy()

            return temporal_crop

        else:
            # if testing , sample a center crop

            start = int((1 - l_ratio[0]) * num_of_frames / 2)
            data = input_data[:, start:num_of_frames - start, :, :]
            temporal_crop_length = data.shape[1]

            temporal_crop = torch.tensor(data, dtype=torch.float)
            temporal_crop = temporal_crop.permute(0, 2, 3, 1).contiguous().view(C * V * M, temporal_crop_length)
            temporal_crop = temporal_crop[None, :, :, None]
            temporal_crop = F.interpolate(temporal_crop, size=(output_size, 1), mode='bilinear', align_corners=False)
            temporal_crop = temporal_crop.squeeze(dim=3).squeeze(dim=0)
            temporal_crop = temporal_crop.contiguous().view(C, V, M, output_size).permute(0, 3, 1,
                                                                                          2).contiguous().numpy()

            return temporal_crop

