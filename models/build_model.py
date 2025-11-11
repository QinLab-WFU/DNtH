# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------
import os
import torch
from .swin_transformer import SwinTransformer
# from .swin_mlp import SwinMLP
# from .RelaHash import RelaHash
from .Nolotransformer import Network
# from .tcdhmodel import TCDHmodule
# from .models import Conformer
# from .navigateNet import attention_net
# from .Net import AlexNet
def build_model(config):
    model_type = config.MODEL.TYPE
    if model_type == 'swin':
        model = SwinTransformer(img_size=config.DATA.IMG_SIZE,
                                patch_size=config.MODEL.SWIN.PATCH_SIZE,
                                in_chans=config.MODEL.SWIN.IN_CHANS,
                                num_classes=config.MODEL.NUM_CLASSES,
                                embed_dim=config.MODEL.SWIN.EMBED_DIM,
                                depths=config.MODEL.SWIN.DEPTHS,
                                num_heads=config.MODEL.SWIN.NUM_HEADS,
                                window_size=config.MODEL.SWIN.WINDOW_SIZE,
                                mlp_ratio=config.MODEL.SWIN.MLP_RATIO,
                                qkv_bias=config.MODEL.SWIN.QKV_BIAS,
                                qk_scale=config.MODEL.SWIN.QK_SCALE,
                                drop_rate=config.MODEL.DROP_RATE,
                                drop_path_rate=config.MODEL.DROP_PATH_RATE,
                                ape=config.MODEL.SWIN.APE,
                                patch_norm=config.MODEL.SWIN.PATCH_NORM,
                                use_checkpoint=config.TRAIN.USE_CHECKPOINT,
                                hash_length=config.MODEL.hash_length)
    # elif model_type == 'swin_mlp':
    #     models = SwinMLP(img_size=config.DATA.IMG_SIZE,
    #                     patch_size=config.MODEL.SWIN_MLP.PATCH_SIZE,
    #                     in_chans=config.MODEL.SWIN_MLP.IN_CHANS,
    #                     num_classes=config.MODEL.NUM_CLASSES,
    #                     embed_dim=config.MODEL.SWIN_MLP.EMBED_DIM,
    #                     depths=config.MODEL.SWIN_MLP.DEPTHS,
    #                     num_heads=config.MODEL.SWIN_MLP.NUM_HEADS,
    #                     window_size=config.MODEL.SWIN_MLP.WINDOW_SIZE,
    #                     mlp_ratio=config.MODEL.SWIN_MLP.MLP_RATIO,
    #                     drop_rate=config.MODEL.DROP_RATE,
    #                     drop_path_rate=config.MODEL.DROP_PATH_RATE,
    #                     ape=config.MODEL.SWIN_MLP.APE,
    #                     patch_norm=config.MODEL.SWIN_MLP.PATCH_NORM,
    #                     use_checkpoint=config.TRAIN.USE_CHECKPOINT)
    # elif model_type == 'RelaHash':
    #     models =RelaHash(nbit = config.MODEL.hash_length,
    #                     nclass=config.MODEL.NUM_CLASSES,
    #                     batchsize=config.DATA.BATCH_SIZE,
    #                     hash_length=config.MODEL.hash_length)
    elif model_type == 'Network':
        model = Network(
                        hash_bit=config.MODEL.hash_length,
                        embed_dim=128,
                        num_classes=config.MODEL.NUM_CLASSES)
    # elif model_type == 'TCDH':
    #     models = TCDHmodule(hash_bit=config.MODEL.hash_length,
    #                         num_classes=config.MODEL.NUM_CLASSES)
    # elif model_type == 'Conformer':
    #     models = Conformer(patch_size=16, channel_ratio=6, embed_dim=576, depth=12, num_heads=9, mlp_ratio=4,
    #                         qkv_bias=True, num_classes=config.MODEL.NUM_CLASSES, hash_length=config.MODEL.hash_length)
    #
    #     if os.path.exists(config.MODEL.PRETRAINED):
    #         print('==> Loading from pretrained models..')
    #         state_dict = torch.load(config.MODEL.PRETRAINED)
    #         state_dict.pop('trans_cls_head.weight')
    #         state_dict.pop('trans_cls_head.bias')
    #         state_dict.pop('conv_cls_head.weight')
    #         state_dict.pop('conv_cls_head.bias')
    #         models.load_state_dict(state_dict, strict=False)
    # elif model_type == 'navigateNet':
    #     models = attention_net(topN=4,
    #                           hash_bit=config.MODEL.hash_length,
    #                           n_class=config.MODEL.NUM_CLASSES
    #                           )
    # elif model_type == 'FAH':
    #     models = AlexNet(num_classes=config.MODEL.NUM_CLASSES,
    #                     Kbits=config.MODEL.hash_length
    #                     )

    else:
        raise NotImplementedError(f"Unkown model: {model_type}")

    return model
