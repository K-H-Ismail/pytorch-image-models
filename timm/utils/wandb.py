import argparse
import torch.distributed as dist
import torch
import pandas as  pd
import plotly.express as px
import tempfile

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0

def str2bool(v):
    """
    Converts string to bool type; enables command line
    arguments in the format of '--arg1 true --arg2 false'
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

class WandbLogger(object):
    def __init__(self, args):
        self.args = args

        try:
            import wandb
            self._wandb = wandb
        except ImportError:
            raise ImportError(
                "To use the Weights and Biases Logger please install wandb."
                "Run `pip install wandb` to install it."
            )

        # Initialize a W&B run
        if self._wandb.run is None:
            self._wandb.init(
                project=args.experiment,
                config=args
            )

    def log_epoch_metrics(self, metrics, commit=True):
        """
        Log train/test metrics onto W&B.
        """
        # Log number of model parameters as W&B summary
        self._wandb.summary['n_parameters'] = metrics.get('n_parameters', None)
        metrics.pop('n_parameters', None)

        # Log current epoch
        self._wandb.log({'epoch': metrics.get('epoch')}, commit=False)
        metrics.pop('epoch')

        for k, v in metrics.items():
            if 'train' in k:
                self._wandb.log({f'Global Train/{k}': v}, commit=False)
            elif 'test' in k:
                self._wandb.log({f'Global Test/{k}': v}, commit=False)

        self._wandb.log({})

    def log_checkpoints(self):
        output_dir = self.args.output_dir
        model_artifact = self._wandb.Artifact(
            self._wandb.run.id + "_model", type="model"
        )

        model_artifact.add_dir(output_dir)
        self._wandb.log_artifact(model_artifact, aliases=["latest", "best"])

    def set_steps(self):
        # Set global training step
        # self._wandb.define_metric('Rank-0 Batch Wise/*', step_metric='Rank-0 Batch Wise/global_train_step')
        # Set epoch-wise step
        self._wandb.define_metric('Global Train/*', step_metric='epoch')
        self._wandb.define_metric('Global Test/*', step_metric='epoch')
        self._wandb.define_metric('Dcls pos |P| max/*', step_metric='epoch')
        self._wandb.define_metric('Dcls pos avg speed/*', step_metric='epoch')
        self._wandb.define_metric('Dcls heatmap hists/*', step_metric='epoch')
        self._wandb.define_metric('Dcls scatters/*', step_metric='epoch')



class DclsVisualizer(object):
    def __init__(self, wandb_logger=None, num_bins=7, epoch=0, dcls_df=None, num_stages = 4, max_epoch=300):
        self.wandb_logger = wandb_logger
        self.num_bins = num_bins
        self.p_prev = {}
        self.num_stages = num_stages
        self.epoch = epoch
        self.max_epoch = max_epoch
        self.df = {}

        if dcls_df is not None:
            self.df = dcls_df

    def log_layer(self, model, stage, block):
        key = 's{stage},b{block}'.format(stage=stage,block=block)
        p = getattr(model.module, 'layer' + str(stage+1))[block].conv2 if hasattr(model, 'module') else getattr(model, 'layer' + str(stage+1))[block].conv2
        out_channels, kernel_count = p.out_channels, p.kernel_count
        p = p.P * p.scaling

        if key not in self.p_prev:
            self.p_prev[key] = torch.zeros_like(p)

        speed = (p - self.p_prev[key]).abs().mean()

        self.wandb_logger._wandb.log({'Dcls pos |P| max/(s{stage},b{block})'.format(stage=stage,block=block): p.abs().max()})
        self.wandb_logger._wandb.log({'Dcls pos avg speed/(s{stage},b{block})'.format(stage=stage,block=block): speed})
        self.wandb_logger._wandb.log({'Dcls heatmap hists/(s{stage},b{block})'.format(stage=stage,block=block): self.wandb_logger._wandb.Histogram(p.detach().cpu(), num_bins=self.num_bins)})

        step = max(out_channels//4, 1)
        p_df = p[:,0:-1:step,0,:].clamp(-self.num_bins//2, self.num_bins//2)
        categories = torch.arange(p_df.size(1)).repeat_interleave(kernel_count)
        df = pd.DataFrame(p_df.reshape((2, -1)).detach().cpu().numpy().T)
        df[2]  = categories
        df[3]  = self.epoch

        self.df[key] = df if key not in self.df else pd.concat([self.df[key], df], ignore_index=True)

        step = 298 #max(self.max_epoch//10, 1)
        if self.epoch % step == 0:
            fig = px.scatter(self.df[key], x=0, y=1, color=2, range_x=[-self.num_bins//2,self.num_bins//2], range_y=[-self.num_bins//2,self.num_bins//2], animation_frame=3, size=2)
            with tempfile.NamedTemporaryFile() as fp:
                fig.write_html(fp.name)
                self.wandb_logger._wandb.log({'Dcls scatters/(s{stage},b{block})'.format(stage=stage,block=block):
                                              self.wandb_logger._wandb.Html(open(fp.name), inject=False)})
        self.p_prev[key] = p

    def log_all_layers(self, model, sync=False):
        for stage in range(self.num_stages):
            if sync:
                self.log_layer(model, stage, 0)
            else:
                for block in range(model.module.layers[stage]):
                    self.log_layer(model, stage, block)
        self.epoch += 1

def get_dcls_loss_rep(model, loss):
    layer_count = 0
    loss_rep = torch.zeros_like(loss)
    for name, param in model.named_parameters():
        print(name)
        if name.endswith(".P"):
            layer_count += 1
            chout, chin, k_count = param.size(1), param.size(2), param.size(3)
            P = param.view(2, chout * chin, k_count)
            P = P.permute(1,2,0).contiguous()
            distances = torch.cdist(P,P,p=2)
            distances_triu = (1-distances).triu(diagonal=1)
            loss_rep += 2*torch.sum(torch.clamp_min(distances_triu , min=0)) / (k_count*(k_count-1)*chout*chin)

    loss_rep /= layer_count
    return loss_rep
