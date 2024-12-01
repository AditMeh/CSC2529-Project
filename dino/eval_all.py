import argparse
from glob import glob
import os
import json

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--method", default="base", type=str, help="base, naive, lora, ewc")
    parser.add_argument("--data_path", default="", required=True, type=str)
    parser.add_argument("--checkpoints_dir", default="", required=True, type=str, help="path to directory with checkpoints to evaluate")
    parser.add_argument("--eval_type", default="knn", required=True, type=str, help="classification head, knn or linear (not implemented yet)")
    parser.add_argument("--arch", default="vit_tiny", required=True, type=str, help="architecture of pretrained model")
    parser.add_argument("--master_port", default="29501", required=True, type=str, help="port for distributed workflow")
    parser.add_argument("--ckpt_path", default=None, required=False, type=str, help="overload for checkpoints_dir if only one checkpoint is to be evaluated")

    args = parser.parse_args()

    if args.ckpt_path is None:
        ckpts = glob(os.path.join(args.checkpoints_dir, "*.pth"))
    else:
        ckpts = [args.ckpt_path]

    eval_splits = ["val", "test", "train", "harmful", "not_harmful"]

    results = {
        "temp": "WO"
    }
    
    filename = "./results_ckpt_500.json"

    if not os.path.exists(f"{filename}"):
        with open(f"{filename}", "w+") as f:
            json.dump(results, f, indent=4)

    for ckpt in ckpts:
        for split in eval_splits:
            cmd = f"python3 -m torch.distributed.launch --nproc_per_node=4 --master_port {args.master_port} eval_{args.eval_type}.py --arch {args.arch} --data_path {args.data_path} --pretrained_weights {ckpt} --json_path {filename} --json_key {args.method} --eval_split {split}"

            os.system(cmd)

        