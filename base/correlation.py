r""" Provides functions that builds/manipulates correlation tensors """
import torch
import torch.nn.functional as F


class Correlation:

    @classmethod
    def multilayer_correlation(cls, query_feats, support_feats, support_mask):
        eps = 1e-5

        corrs = []
        for idx, (query_feat, support_feat) in enumerate(zip(query_feats, support_feats)):
            bsz, ch, hb, wb = support_feat.size()
            support_mask_s = F.interpolate(support_mask, size=(hb, wb), mode='bilinear', align_corners=True)
            support_feat = support_feat * support_mask_s
            support_feat = support_feat.view(bsz, ch, -1)
            support_feat = support_feat / (support_feat.norm(dim=1, p=2, keepdim=True) + eps)

            bsz, ch, ha, wa = query_feat.size()
            query_feat = query_feat.view(bsz, ch, -1)
            query_feat = query_feat / (query_feat.norm(dim=1, p=2, keepdim=True) + eps)

            corr = torch.bmm(query_feat.transpose(1, 2), support_feat)
            corr = corr.clamp(min=0)
            corrs.append(corr)

        corrs = torch.stack(corrs, dim=1)

        return corrs