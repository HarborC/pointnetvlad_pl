import os
import torch

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from datasets.UgvDataset import UgvDataset

# models
from models.pointnetvlad import PointNetVlad

# optimizer, scheduler, visualization
from utils import *

# losses
from losses import quadruplet_loss

# metrics
from metrics import *

# pytorch-lightning
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TestTubeLogger

from option import get_options

torch.backends.cudnn.benchmark = True # this increases training speed by 5x

class System(pl.LightningModule):
    def __init__(self, hparams):
        super(System, self).__init__()
        self.save_hyperparameters(hparams)

        self.loss = quadruplet_loss.QuadrupletLoss(self.hparams.margin_1, self.hparams.margin_2, \
                                                   use_min=self.hparams.triplet_use_best_positives, \
                                                   lazy=self.hparams.loss_lazy,
                                                   ignore_zero_loss=self.hparams.loss_ignore_zero_batch)
        self.model = PointNetVlad(self.hparams.num_points, output_dim=self.hparams.output_dim, emb_dims=self.hparams.emb_dims)

        # if num gpu is 1, print model structure and number of params
        if self.hparams.num_gpus == 1:
            # print(self.model)
            print('number of parameters : %.2f M' % 
                  (sum(p.numel() for p in self.model.parameters() if p.requires_grad) / 1e6))
        
        # load model if checkpoint path is provided
        if self.hparams.ckpt_path != '':
            print('Load model from', self.hparams.ckpt_path)
            load_ckpt(self.model, self.hparams.ckpt_path, self.hparams.prefixes_to_ignore)

    def forward(self, query, positives, negatives, other_neg):
        # print("query: ", query.shape)
        # print("positives: ", positives.shape)
        # print("negatives: ", negatives.shape)
        # print("other_neg: ", other_neg.shape)
        feed_tensor = torch.cat((query, positives, negatives, other_neg), 1)
        feed_tensor = feed_tensor.view((-1, 1, self.hparams.num_points, 3))
        feed_tensor = feed_tensor.cuda()

        output = self.model(feed_tensor)

        output = output.view(self.hparams.batch_size, -1, self.hparams.output_dim)
        o1, o2, o3, o4 = torch.split(
            output, [1, self.hparams.positives_per_query, self.hparams.negatives_per_query, 1], dim=1)
        return o1, o2, o3, o4

    def training_step(self, batch, batch_idx):
        query, positives, negatives, other_neg = batch
        output_query, output_positives, output_negatives, output_other_neg = self.forward(query, positives, negatives, other_neg)
        loss = self.loss(output_query, output_positives, output_negatives, output_other_neg)

        return {
                'loss': loss,
                'log': {'train/loss': loss,
                        'lr': get_learning_rate(self.optimizer)}
               }

    def validation_step(self, batch, batch_idx):
        query, positives, negatives, other_neg = batch

        with torch.no_grad():
            output_query, output_positives, output_negatives, output_other_neg = self.forward(query, positives, negatives, other_neg)
            loss = self.loss(output_query, output_positives, output_negatives, output_other_neg)

        return {'val_loss': loss}

    def validation_end(self, outputs):
        mean_loss = torch.stack([x['val_loss'] for x in outputs]).mean()

        return {
                'progress_bar': {'val_loss': mean_loss},
                'log': {'val/loss': mean_loss}
               }

    def configure_optimizers(self):
        self.optimizer = get_optimizer(self.hparams, self.model)
        scheduler = get_scheduler(self.hparams, self.optimizer)
        
        return [self.optimizer], [scheduler]

    def train_dataloader(self):
        train_dataset = UgvDataset(self.hparams, type='train')
        if self.hparams.num_gpus > 1:
            sampler = DistributedSampler(train_dataset)
        else:
            sampler = None
        return DataLoader(train_dataset, 
                          shuffle=(sampler is None),
                          sampler=sampler,
                          num_workers=4,
                          batch_size=self.hparams.batch_size,
                          pin_memory=True)

    def val_dataloader(self):
        val_dataset = UgvDataset(self.hparams, type='val')
        if self.hparams.num_gpus > 1:
            sampler = DistributedSampler(val_dataset)
        else:
            sampler = None
        return DataLoader(val_dataset, 
                          shuffle=(sampler is None),
                          sampler=sampler,
                          num_workers=4,
                          batch_size=self.hparams.batch_size,
                          pin_memory=True)


if __name__ == '__main__':
    hparams = get_options()
    system = System(hparams)
    checkpoint_callback = ModelCheckpoint(filename=os.path.join(f'ckpts/{hparams.exp_name}',
                                                                '{epoch:02d}'),
                                          monitor='val/loss',
                                          mode='min',
                                          save_top_k=1,
                                          dirpath="/media/s1/cjg/GRP/PointNetVlad/pointnetvlad_pl/log")

    logger = TestTubeLogger(
        save_dir="logs",
        name=hparams.exp_name,
        debug=False,
        create_git_tag=False
    )

    trainer = Trainer(max_epochs=hparams.num_epochs,
                      checkpoint_callback=checkpoint_callback,
                      logger=logger,
                      weights_summary=None,
                      progress_bar_refresh_rate=1,
                      gpus=hparams.num_gpus,
                      strategy="ddp_sharded" if hparams.num_gpus>1 else None,
                      num_sanity_val_steps=0 if hparams.num_gpus>1 else 5,
                      benchmark=True,
                      precision=16 if hparams.use_amp else 32,
                      amp_level='O1',
                      amp_backend='apex',
                      accelerator='gpu')

    trainer.fit(system)