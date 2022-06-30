# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import argparse
import os
import uuid
from pathlib import Path

import train as classification
import submitit

def parse_args():
    classification_parser, config_parser = classification.get_args_parser()
    parser = argparse.ArgumentParser("Submitit for Vision classification references", parents=[classification_parser])
    parser.add_argument('-c', '--config', default='', type=str, metavar='FILE',
                        help='YAML config file specifying default arguments')
    group = parser.add_argument_group("Submitit parameters")
    group.add_argument("--ngpus", default=8, type=int, help="Number of gpus to request on each node")
    group.add_argument("--nodes", default=1, type=int, help="Number of nodes to request")
    group.add_argument("--timeout", default=20, type=int, help="Duration of the job, in hours")
    group.add_argument("--job_name", default="resnet", type=str, help="Job name")
    group.add_argument("--job_dir", default="", type=str, help="Job directory; leave empty for default")
    group.add_argument("--partition", default="gpu_p2", type=str, help="Partition where to submit")
    group.add_argument("--use_volta32", action='store_true', default=False, help="Big models? Use this")
    group.add_argument("--constraint", default="v100", type=str, help="constraint v100 or a100")
    group.add_argument("--account", default="owj@v100", type=str, help="Account name")
    group.add_argument('--comment', default="", type=str,
                        help='Comment to pass to scheduler, e.g. priority message')

    # Do we have a config file to parse?
    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, 'r') as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)

    # The main arg parser parses the rest of the args, the usual
    # defaults will have been overridden if config file specified.
    args = parser.parse_args(remaining)

    # Cache the args as a text string to save them in the output dir later
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    return args, args_text

def get_shared_folder() -> Path:
    work = os.getenv("WORK")
    if Path(f"{work}/pytorch-image-models/checkpoint").is_dir():
        p = Path(f"{work}/pytorch-image-models/checkpoint")
        p.mkdir(exist_ok=True)
        return p
    raise RuntimeError("No shared folder available")

def get_init_file():
    # Init file must not exist, but it's parent dir must exist.
    os.makedirs(str(get_shared_folder()), exist_ok=True)
    init_file = get_shared_folder() / f"{uuid.uuid4().hex}_init"
    if init_file.exists():
        os.remove(str(init_file))
    return init_file

class Trainer(object):
    def __init__(self, args):
        self.args = args

    def __call__(self):
        import train as classification

        self._setup_gpu_args()
        classification.main(self.args)

    def checkpoint(self):
        import os
        import submitit

        self.args.dist_url = get_init_file().as_uri()
        self.args.auto_resume = True
        print("Requeuing ", self.args)
        empty_trainer = type(self)(self.args)
        return submitit.helpers.DelayedSubmission(empty_trainer)

    def _setup_gpu_args(self):
        import submitit
        from pathlib import Path

        job_env = submitit.JobEnvironment()
        self.args.output_dir = Path(self.args.job_dir)
        self.args.gpu = job_env.local_rank
        self.args.rank = job_env.global_rank
        self.args.world_size = job_env.num_tasks
        print(f"Process group: {job_env.num_tasks} tasks, rank: {job_env.global_rank}")


def main():
    args, args_text = parse_args()
    if args.job_dir == "":
        args.job_dir = get_shared_folder() / "%j"

    executor = submitit.AutoExecutor(folder=args.job_dir, slurm_max_num_timeout=30)

    num_gpus_per_node = args.ngpus
    nodes = args.nodes
    timeout_min = args.timeout * 60

    partition = args.partition
    kwargs = {}
    if args.use_volta32:
        kwargs['slurm_constraint'] = 'v100-32g'
    if args.comment:
        kwargs['slurm_comment'] = args.comment

    kwargs['slurm_constraint'] = args.constraint

    executor.update_parameters(
        #mem_gb=40 * num_gpus_per_node,
        gpus_per_node=num_gpus_per_node,
        tasks_per_node=num_gpus_per_node,  # one task per GPU
        cpus_per_task=args.workers,
        slurm_account=args.account,
        nodes=nodes,
        timeout_min=timeout_min,  # max is 60 * 72
        # Below are cluster dependent parameters
        slurm_partition=partition,
        slurm_signal_delay_s=120,
        **kwargs
    )

    executor.update_parameters(name=args.job_name)

    args.dist_url = get_init_file().as_uri()
    args.output_dir = args.job_dir

    trainer = Trainer(args, args_text)
    job = executor.submit(trainer)

    print("Submitted job_id:", job.job_id)

if __name__ == "__main__":
    main()

