import numpy as np
import torch
import os


class ABDChecker(object):

    def __init__(self, window_width, shift, feat_dir=None, feat_size=None):
        self.ABD = window_width
        self.frames_seen = 0
        if feat_dir is not None:
            self.dirname = feat_dir + shift
            self.last_i3d_features = torch.zeros((self.ABD, 1024))
        else:
            self.dirname = None
            self.last_i3d_features = torch.zeros((self.ABD, feat_size))
        self.last_smooth_i3d_feat = None
        self.similarities = []

    def compute_similarity(self, i_val, i3d_feat=None):
        self.frames_seen += 1
        if self.dirname is not None:
            i3d_feat = torch.load(os.path.join(self.dirname, str(i_val) + ".pt")).detach().cpu()
        self.last_i3d_features = torch.cat((self.last_i3d_features[1:, :], i3d_feat), dim=0)
        smooth_i3d_feat = torch.mean(self.last_i3d_features, dim=0)
        if self.frames_seen > self.ABD:
            self.similarities.append(torch.nn.CosineSimilarity(dim=0)(smooth_i3d_feat,
                                                                      self.last_smooth_i3d_feat))
        self.last_smooth_i3d_feat = smooth_i3d_feat

    def check_reset(self):
        if self.frames_seen < self.ABD * 2 or self.frames_seen % self.ABD != 0:
            return False
        threshold = np.percentile(self.similarities, 25, interpolation="midpoint")
        window_min = min(self.similarities[-self.ABD:])
        if window_min < threshold:
            return False
        else:
            return True
