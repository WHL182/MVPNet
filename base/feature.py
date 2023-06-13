r""" Extracts intermediate features from given backbone network & layer ids """
import torch.nn.functional as F


def extract_feat_vgg(img, backbone, feat_ids, bottleneck_ids=None, lids=None, pool=False, pool_thr=50):
    r""" Extract intermediate features from VGG """
    feats = []
    feat = img
    for lid, module in enumerate(backbone.features):
        feat = module(feat)
        if lid in feat_ids:
            if pool and feat.shape[-1] >= pool_thr:
                    feats.append(F.avg_pool2d(feat.clone(), kernel_size=3, stride=2, padding=1))
            else:
                feats.append(feat.clone())
    return feats


def extract_feat_res(img, backbone, feat_ids, bottleneck_ids, lids, pool=False, pool_thr=50):
    r""" Extract intermediate features from ResNet"""
    feats = []

    # Layer 0
    feat = backbone.layer0[0].forward(img)
    feat = backbone.layer0[1].forward(feat)
    feat = backbone.layer0[2].forward(feat)
    feat = backbone.layer0[3].forward(feat)
    feat = backbone.layer0[4].forward(feat)
    feat = backbone.layer0[5].forward(feat)
    feat = backbone.layer0[6].forward(feat)
    feat = backbone.layer0[7].forward(feat)
    feat = backbone.layer0[8].forward(feat)
    feat = backbone.layer0[9].forward(feat)

    # Layer 1-4
    for hid, (bid, lid) in enumerate(zip(bottleneck_ids, lids)):
        res = feat
        feat = backbone.__getattr__('layer%d' % lid)[bid].conv1.forward(feat)
        feat = backbone.__getattr__('layer%d' % lid)[bid].bn1.forward(feat)
        feat = backbone.__getattr__('layer%d' % lid)[bid].relu.forward(feat)
        feat = backbone.__getattr__('layer%d' % lid)[bid].conv2.forward(feat)
        feat = backbone.__getattr__('layer%d' % lid)[bid].bn2.forward(feat)
        feat = backbone.__getattr__('layer%d' % lid)[bid].relu.forward(feat)
        feat = backbone.__getattr__('layer%d' % lid)[bid].conv3.forward(feat)
        feat = backbone.__getattr__('layer%d' % lid)[bid].bn3.forward(feat)

        if bid == 0:
            res = backbone.__getattr__('layer%d' % lid)[bid].downsample.forward(res)

        feat += res

        if hid + 1 in feat_ids:
            if pool and feat.shape[-1] >= pool_thr:
                feats.append(F.avg_pool2d(feat.clone(), kernel_size=3, stride=2, padding=1))
            else:
                feats.append(feat.clone())

        feat = backbone.__getattr__('layer%d' % lid)[bid].relu.forward(feat)

    return feats