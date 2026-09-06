from .nets import *
# from .PVTv2 import *
# from .medVit import *

from modeling.VMamba.OCAB_VMamba import *
def get_net(cfg):
    if cfg.ALGORITHM == 'ERM' or cfg.ALGORITHM == 'GDRNet':
        net = get_backbone(cfg)
    elif cfg.ALGORITHM == 'GREEN':
        net = SoftLabelGCN(cfg)
    elif cfg.ALGORITHM == 'CABNet':
        net = CABNet(cfg)
    elif cfg.ALGORITHM == 'MixupNet':
        net = MixupNet(cfg)
    elif cfg.ALGORITHM == 'MixStyleNet':
        net = MixStyleNet(cfg)
    elif cfg.ALGORITHM == 'Fishr' or cfg.ALGORITHM == 'DRGen':
        net = FishrNet(cfg)
    else:
        raise ValueError('Wrong type')
    return net


def flatten_dict(nested_dict, parent_key='', sep='.'):
    """
    Ví dụ: {"layers": {"0": {"ocab": tensor}}} -> {"layers.0.ocab": tensor}
    """
    items = []
    for k, v in nested_dict.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k

        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))

    return dict(items)

def get_backbone(cfg):
    if cfg.BACKBONE == 'resnet18':
        model = resnet18(pretrained=True)
    elif cfg.BACKBONE == 'resnet50':
        model = resnet50(pretrained=True)
    elif cfg.BACKBONE == 'resnet101':
        model = resnet101(pretrained=True)
    # elif cfg.BACKBONE == "pvt":
    #     model = pvt_v2_b2()
    #     checkpoint = torch.load("pvt_v2_b2.pth", map_location='cpu')
    #
    #     if 'model' in checkpoint:
    #         checkpoint_model = checkpoint['model']
    #     else:
    #         checkpoint_model = checkpoint
    #     state_dict = model.state_dict()
    #     for k in ['head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias', '']:
    #         if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
    #             print(f"Removing key {k} from pretrained checkpoint")
    #             del checkpoint_model[k]
    #
    #     model.load_state_dict(checkpoint_model, strict=False)
    # elif cfg.BACKBONE == 'medvit':
    #     model = MedViT_small()
    #     checkpoint = torch.load("MedViT_small_im1k.pth", map_location='cpu')
    #     if 'model' in checkpoint:
    #         checkpoint_model = checkpoint['model']
    #     else:
    #         checkpoint_model = checkpoint
    #     state_dict = model.state_dict()
    #     for k in ['proj_head.0.weight', 'proj_head.0.bias']:
    #         if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
    #             print(f"Removing key {k} from pretrained checkpoint")
    #             del checkpoint_model[k]
    #     model.load_state_dict(checkpoint_model, strict=False)

    # elif cfg.BACKBONE == 'DGMamba':
    #     model = build_vssm_model(config=cfg,is_pretrain=True)
    elif cfg.BACKBONE == 'VMamba':
        model = vmamba_small_m2()
        checkpoint = torch.load("/home/ai3/NTBao1/DGDR/modeling/VMamba/vssm_small_0229_ckpt_epoch_222.pth", map_location='cpu')

        if 'model' in checkpoint:
            checkpoint_model = checkpoint['model']
        else:
            checkpoint_model = checkpoint

        model_dict = model.state_dict()
        new_checkpoint = {}

        for k, v in checkpoint_model.items():


            if k.startswith("classifier.head"):
                print(f"Skip {k} (classifier)")
                continue


            if k in model_dict:
                if v.shape == model_dict[k].shape:
                    new_checkpoint[k] = v
                else:
                    print(f"Skip {k}: {v.shape} != {model_dict[k].shape}")
            else:
                print(f"Drop key {k} (not in model)")

        # update vào model
        model_dict.update(new_checkpoint)
        model.load_state_dict(new_checkpoint, strict=False)
    else:
        raise ValueError('Wrong type')

    return model

def get_classifier(out_feature_size, cfg):
    return torch.nn.Linear(out_feature_size, cfg.DATASET.NUM_CLASSES)