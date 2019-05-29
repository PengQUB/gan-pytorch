# -*- coding: utf-8 -*-
import json
import os
import logging
import torch
from torch import optim, nn
from torch.autograd import Variable
from models import ModelSelector
from datasets import get_loader
from utils import AveMeter, Timer, patch_replication_callback
from utils.visualization import vis_seq
from tensorboardX import SummaryWriter
from torchvision.utils import make_grid
from models import Generator, Discriminator

logger = logging.getLogger('InfoLog')

class Trainer(object):
    def __init__(self, config):
        self.config = config
        self.ckpt_dir = config.ckpt_dir
        if not os.path.exists(self.ckpt_dir):
            os.makedirs(self.ckpt_dir)

        self.save_config(config)
        self.timer = Timer()

        self.writer = SummaryWriter(log_dir=config.ckpt_dir)

        self.lr = config.lr
        self.datasets, self.loaders = get_loader(config)
        self.max_iters = config.max_iters
        if self.max_iters is not None:
            self.epochs = self.max_iters // len(self.loaders['train'])
        else:
            self.epochs = config.epochs
        self.start_epoch = 0

        # self.scores = ScoreMeter(self.num_classes)

        ### Network ###
        self.netG_A2B = ModelSelector[config.G_model].ResnetGenerator(input_nc = config.in_channels,
                                                         output_nc = config.out_channels,
                                                         use_dropout = config.use_dropout,
                                                         **config.model_params[config.G_model])
        self.netG_B2A = ModelSelector[config.G_model].ResnetGenerator(input_nc = config.in_channels,
                                                         output_nc = config.out_channels,
                                                         use_dropout = config.use_dropout,
                                                         **config.model_params[config.G_model])
        self.netD_A = Discriminator(input_nc = config.in_channels)
        self.netD_B = Discriminator(input_nc = config.out_channels)

        self.criterion_GAN = nn.MSELoss()
        self.criterion_cycle = nn.L1Loss()
        self.criterion_identity = nn.L1Loss()

        if config.distributed:
            self.netG_A2B = nn.DataParallel(self.netG_A2B)
            self.netG_B2A = nn.DataParallel(self.netG_B2A)
            self.netD_A = nn.DataParallel(self.netD_A)
            self.netD_B = nn.DataParallel(self.netD_B)
            patch_replication_callback(self.netG_A2B)
            patch_replication_callback(self.netG_B2A)
            patch_replication_callback(self.netD_A)
            patch_replication_callback(self.netD_B)

        if self.config.cuda:
            self.model = self.model.cuda()
            self.criterion = self.criterion.cuda()

        # self.criterion = LossSelector[config.loss](**config.loss_params[config.loss])
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, betas=(0.9, 0.999), eps=1e-8)
        if self.max_iters is not None:
            self.lr_decay = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, self.max_iters)
        elif self.epochs is None:
            self.lr_decay = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, self.epochs*len(self.loaders['train']))

        self.best_loss = float('inf')

        if config.resume:
            logger.info('***Resume from checkpoint***')
            state = torch.load(os.path.join(self.ckpt_dir, 'ckpt.pt'))
            self.model.load_state_dict(state['net'])
            self.start_epoch = state['epoch']
            self.best_loss = state['best_loss']
            self.optimizer.load_state_dict(state['optim'])
            self.lr_decay.load_state_dict(state['lr_decay'])
            self.lr_decay.last_epoch = self.start_epoch

    def train_and_val(self):
        for epoch in range(self.start_epoch, self.epochs):
            logger.info(f"Epoch :[{epoch}/{self.epochs}]")
            self.train(epoch)
            logger.info(f"val starts...")
            val_loss = self.val(epoch)
            if val_loss < self.best_loss:
                logger.info('------')
                self.best_loss = val_loss
                self.save({'net': self.model.state_dict(),
                           'best_loss': val_loss,
                           'epoch': epoch,
                           'optim': self.optimizer.state_dict(),
                           'lr_decay': self.lr_decay.state_dict()})
        logger.info(f"best val loss: {self.best_loss}")

        self.writer.close()

    def train(self, epoch):
        losses = AveMeter()
        # self.scores.reset()

        self.model.train()
        for i, (input_A, input_B) in enumerate(self.loaders['train']):
            self.lr_decay.step()

            if self.config.cuda:
                input_A = Variable(input_A).cuda()
                input_B = Variable(input_B).cuda()

            outs = self.model(imgs, neighbors, flows)

            if self.config.residual:
                outs = outs + bicubics

            loss = self.criterion(outs, targets)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            losses.update(loss.item(), imgs.size()[0])
            # scores, _ = self.scores.get_scores()

            if i % 200 == 0 or i == len(self.loaders['train']) - 1:
                logger.info(f"Train: [{i}/{len(self.loaders['train'])}] | "
                            f"Time: {self.timer.timeSince()} | "
                            f"loss: {losses.avg:.4f} | "
                            # f"oa:{scores['oa']:.4f} | "
                            # f"ma: {scores['ma']:.4f} | "
                            # f"fa: {scores['fa']:.4f} | "
                            # f"miou: {scores['miou']:.4f}"
                )

        self.writer.add_scalar('train/loss', losses.avg, epoch)
        # self.writer.add_scalar('train/mIoU', scores['miou'], epoch)
        # self.writer.add_scalar('train/Aacc', scores['oa'], epoch)
        # self.writer.add_scalar('train/Acc_class', scores['ma'], epoch)
        # self.writer.add_scalar('train/Acc_freq', scores['fa'], epoch)

    def val(self, epoch):
        losses = AveMeter()
        # self.scores.reset()

        self.model.eval()
        with torch.no_grad():
            for i, (imgs, targets, neighbors, flows, bicubics) in enumerate(self.loaders['val']):
                if self.config.cuda:
                    imgs = Variable(imgs).cuda()
                    targets = Variable(targets).cuda()
                    bicubics = Variable(bicubics).cuda()
                    neighbors = [Variable(j).cuda() for j in neighbors]
                    flows = [Variable(j).cuda().float() for j in flows]

                outs = self.model(imgs, neighbors, flows)

                loss = self.criterion(outs, targets)

                # self.scores.update(targets.cpu().data.numpy(),
                #                    outs.argmax(dim=1).cpu().data.numpy())
                losses.update(loss.item(), imgs.size()[0])

            # scores, cls_iou = self.scores.get_scores()

            logger.info(f"Val: [{i}/{len(self.loaders['val'])}] | "
                        f"Time: {self.timer.timeSince()} | "
                        f"loss: {losses.avg:.4f} | "
                        # f"oa:{scores['oa']:.4f} | "
                        # f"ma: {scores['ma']:.4f} | "
                        # f"fa: {scores['fa']:.4f} | "
                        # f"miou: {scores['miou']:.4f}"
            )

        self.writer.add_scalar('val/loss', losses.avg, epoch)
        # self.writer.add_scalar('val/mIoU', scores['miou'], epoch)
        # self.writer.add_scalar('val/Acc', scores['oa'], epoch)
        # self.writer.add_scalar('val/Acc_class', scores['ma'], epoch)
        # self.writer.add_scalar('val/Acc_freq', scores['fa'], epoch)

        if epoch % 10 == 0:
            self.summary_imgs(imgs, targets, outs, epoch)

        return losses.avg

    def save(self, state):
        torch.save(state, os.path.join(self.ckpt_dir, 'ckpt.pt'))
        logger.info('***Saving model***')

    def save_config(self, config):
        with open(os.path.join(self.ckpt_dir, 'config.json'), 'w+') as f:
            f.write(json.dumps(config.__dict__, indent=4))
        f.close()

    def summary_imgs(self, imgs, targets, outputs, epoch):
        grid_imgs = make_grid(imgs[:3].clone().cpu().data, nrow=3, normalize=True)
        self.writer.add_image('Image', grid_imgs, epoch)
        grid_imgs = make_grid(vis_seq(outputs[:3].cpu().data.numpy()),
                              nrow=3, normalize=False, range=(0, 255))
        self.writer.add_image('Predicted SRimg', grid_imgs, epoch)
        grid_imgs = make_grid(vis_seq(targets[:3].cpu().data.numpy()),
                              nrow=3, normalize=False, range=(0, 255))
        self.writer.add_image('GT SRimg', grid_imgs, epoch)
