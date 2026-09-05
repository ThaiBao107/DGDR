import argparse
from configs.defaults import _C as cfg_default


def get_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--root", type=str, default="../DGDATA/", help="path to dataset")
    parser.add_argument("--root", type=str, default="../data/FundusDG", help="path to dataset")

    parser.add_argument("--algorithm", type=str, default='GDRNet', help='check in algorithms.py')
    # TODO : sua default = swin
    parser.add_argument("--backbone", type=str, default="swin")
    parser.add_argument("--source-domains", type=str, nargs="+", help="source domains for DGDR", default="APTOS")
    parser.add_argument("--target-domains", type=str, nargs="+", help="target domains for DGDR", default="DEEPDR")
    parser.add_argument("--dg_mode", type=str, default='DG', help="DG or ESDG")

    parser.add_argument("--num_classes", type=int, default=5)
    parser.add_argument("--random", action="store_true")
    parser.add_argument("--val_ep", type=int, default=10)
    parser.add_argument("--output", type=str, default='test')
    parser.add_argument("--override", action="store_true")

    return parser.parse_args()


def setup_cfg(args):
    cfg = cfg_default.clone()
    cfg.RANDOM = args.random
    cfg.OUTPUT_PATH = args.output
    cfg.OVERRIDE = args.override
    cfg.DG_MODE = args.dg_mode

    cfg.ALGORITHM = args.algorithm
    cfg.BACKBONE = args.backbone

    cfg.DATASET.ROOT = args.root
    cfg.DATASET.SOURCE_DOMAINS = args.source_domains
    cfg.DATASET.TARGET_DOMAINS = args.target_domains
    # cfg.DATASET.SOURCE_DOMAINS = "APTOS"
    # cfg.DATASET.TARGET_DOMAINS = "DEEPDR"
    cfg.DATASET.NUM_CLASSES = args.num_classes
    cfg.VAL_EPOCH = args.val_ep

    # TODO: Add param to swin
    cfg.IMG_SIZE = 256
    cfg.MODEL.NUM_CLASSES = 5

    # TODO: Add file config to swin
    # if args.backbone == 'swin':
    #     cfg.merge_from_file("./configs/swin_base_patch4_window7_224.yaml")

    if args.backbone == 'swin':
        cfg.merge_from_file("./configs/swin_small_patch4_window7_224_22k.yaml")
    if args.dg_mode == 'DG':
        cfg.merge_from_file("./configs/datasets/GDRBench.yaml")
    elif args.dg_mode == 'ESDG':
        cfg.merge_from_file("./configs/datasets/GDRBench_ESDG.yaml")
    else:
        raise ValueError('Wrong type')

    return cfg
