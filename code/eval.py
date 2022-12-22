# python imports
import argparse
import os
import glob
import time
from pprint import pprint

# torch imports
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data
import torchvision
from torchvision.models.detection import FCOS
from torchvision.models.detection.anchor_utils import AnchorGenerator

# our code
from libs import load_config, build_dataset, build_dataloader, evaluate


################################################################################
def main(args):
    """main function that handles inference"""

    """1. load config"""
    # sanity check
    if os.path.isfile(args.config):
        cfg = load_config(args.config)
    else:
        raise ValueError("Config file does not exist.")
    assert len(cfg["dataset"]["test"]) > 0, "Test set must be specified!"
    """
    if ".pth.tar" in args.ckpt:
        assert os.path.isfile(args.ckpt), "CKPT file does not exist!"
        ckpt_file = args.ckpt
    else:
        assert os.path.isdir(args.ckpt), "CKPT file folder does not exist!"
        if args.epoch > 0:
            ckpt_file = os.path.join(
                args.ckpt, "epoch_{:03d}.pth.tar".format(args.epoch)
            )
        else:
            ckpt_file_list = sorted(glob.glob(os.path.join(args.ckpt, "*.pth.tar")))
            ckpt_file = ckpt_file_list[-1]
        assert os.path.exists(ckpt_file)
    """

    pprint(cfg)

    """2. create dataset / dataloader"""
    val_dataset = build_dataset(
        cfg["dataset"]["name"],
        cfg["dataset"]["test"],
        cfg["dataset"]["img_folder"],
        cfg["dataset"]["json_folder"],
    )
    val_loader = build_dataloader(val_dataset, False, **cfg["loader"])

    """3. create model"""
    # model
    # model = FCOS(**cfg["model"]).to(torch.device(cfg["devices"][0]))

    # load a pre-trained model for classification and return
    # only the features
    backbone = torchvision.models.mobilenet_v2(weights="DEFAULT").features
    # FCOS needs to know the number of
    # output channels in a backbone. For mobilenet_v2, it's 1280
    # so we need to add it here
    backbone.out_channels = 1280

    # let's make the network generate 5 x 3 anchors per spatial
    # location, with 5 different sizes and 3 different aspect
    # ratios. We have a Tuple[Tuple[int]] because each feature
    # map could potentially have different sizes and
    # aspect ratios
    anchor_generator = AnchorGenerator(
        sizes=((8,), (16,), (32,), (64,), (128,)),
        aspect_ratios=((1.0,),)
    )

    # put the pieces together inside a FCOS model
    model = FCOS(
        backbone,
        num_classes=80,
        anchor_generator=anchor_generator,
    )




    # also disable cudnn benchmark, as the input size varies during inference
    cudnn.benchmark = False

    """4. load ckpt"""
    """
    print("=> loading checkpoint '{}'".format(ckpt_file))
    # load ckpt
    checkpoint = torch.load(
        ckpt_file, map_location=lambda storage, loc: storage.cuda(cfg["devices"][0])
    )
    print("Loading model ...")
    model.load_state_dict(checkpoint["state_dict"])
    del checkpoint
    """

    # set up evaluator
    gt_json_file = os.path.join(
        cfg["dataset"]["json_folder"], cfg["dataset"]["test"] + ".json"
    )
    output_file = os.path.join("..","logs", "eval_results.json")

    """5. Test the model"""
    print("\nStart testing ...")
    start = time.time()
    evaluate(
        val_loader,
        model,
        output_file,
        gt_json_file,
        torch.device(cfg["devices"][0]),
        print_freq=args.print_freq,
    )
    end = time.time()
    print("All done! Total time: {:0.2f} sec".format(end - start))
    return


################################################################################
if __name__ == "__main__":
    """Entry Point"""
    # the arg parser
    parser = argparse.ArgumentParser(
        description="Train a point-based transformer for action localization"
    )
    parser.add_argument("config", type=str, metavar="DIR", help="path to a config file")
    parser.add_argument("ckpt", type=str, metavar="DIR", help="path to a checkpoint")
    parser.add_argument("-epoch", type=int, default=-1, help="checkpoint epoch")
    parser.add_argument(
        "-p",
        "--print-freq",
        default=10,
        type=int,
        help="print frequency (default: 10 iterations)",
    )
    args = parser.parse_args()
    main(args)
