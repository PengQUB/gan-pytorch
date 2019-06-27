# -*- coding: utf-8 -*-
import json
import os
import logging
import torch
import itertools
from torch import optim, nn
from torch.autograd import Variable
from models import GModelSelector, DModelSelector
from datasets import get_loader
from utils import AveMeter, Timer, patch_replication_callback, ReplayBuffer
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
        self.lr_decay_start = config.lr_decay_start
        self.datasets, self.loaders = get_loader(config)
        self.max_iters = config.max_iters
        if self.max_iters is not None:
            self.epochs = self.max_iters // len(self.loaders['train'])
        else:
            self.epochs = config.epochs
        self.start_epoch = 0

        ### Network ###
        self.netG = GModelSelector[config.G_model].ResnetGenerator(input_nc = config.in_channels,
                                                         output_nc = config.out_channels,
                                                         use_dropout = config.use_dropout,
                                                         **config.model_params[config.G_model])
        self.netD = DModelSelector[config.D_model].NLayerDiscriminator(input_nc=config.out_channels,
                                                                         n_layers=3)

        self.criterion_GAN = nn.MSELoss()
        self.criterion_L1 = nn.L1Loss()

        if config.distributed:
            self.netG = nn.DataParallel(self.netG)
            self.netD = nn.DataParallel(self.netD)
            patch_replication_callback(self.netG)
            patch_replication_callback(self.netD)

        if self.config.cuda:
            self.netG = self.netG.cuda()
            self.netD = self.netD.cuda()
            self.criterion_GAN = self.criterion_GAN.cuda()
            self.criterion_L1 = self.criterion_L1.cuda()

        # self.criterion = LossSelector[config.loss](**config.loss_params[config.loss])
        self.optimizer_G = optim.Adam(self.netG.parameters(), lr=self.lr, betas=(0.9, 0.999), eps=1e-8)
        self.optimizer_D = optim.Adam(self.netD.parameters(), lr=self.lr, betas=(0.9, 0.999), eps=1e-8)
        if self.max_iters is not None:
            self.lr_decay_G = optim.lr_scheduler.CosineAnnealingLR(self.optimizer_G, self.max_iters)
            self.lr_decay_D = optim.lr_scheduler.CosineAnnealingLR(self.optimizer_D, self.max_iters)
        elif self.epochs is not None:
            self.lr_decay_G = optim.lr_scheduler.LambdaLR(self.optimizer_G, lr_lambda=self.lambda_rule)
            self.lr_decay_D = optim.lr_scheduler.LambdaLR(self.optimizer_D, lr_lambda=self.lambda_rule)
        else:
            raise NotImplementedError('max_iters or epochs cannot be {}'.format(self.epochs))

        self.best_loss = float('inf')

        if config.resume:
            logger.info('***Resume from checkpoint***')
            state = torch.load(os.path.join(self.ckpt_dir, 'netG.pt'))
            self.netG.load_state_dict(state['net'])
            self.start_epoch = state['epoch']
            self.best_loss = state['best_loss']
            self.optimizer_G.load_state_dict(state['optim'])
            self.lr_decay_G.load_state_dict(state['lr_decay_G'])
            self.lr_decay_G.last_epoch = self.start_epoch

            state = torch.load(os.path.join(self.ckpt_dir, 'netD.pt'))
            self.netD.load_state_dict(state['net'])
            self.start_epoch = state['epoch']
            self.best_loss = state['best_loss']
            self.optimizer_D.load_state_dict(state['optim'])
            self.lr_decay_D.load_state_dict(state['lr_decay_D'])
            self.lr_decay_D.last_epoch = self.start_epoch


    def train_and_val(self):
        for epoch in range(self.start_epoch, self.epochs):
            logger.info(f"Epoch :[{epoch}/{self.epochs}]")
            self.train(epoch)
            logger.info(f"val starts...")
            val_loss_G, val_loss_D = self.val(epoch)
            if val_loss_G < self.best_loss:
                logger.info('---Saving Best---')
                self.best_loss = val_loss_G
                self.save({'net': self.netG.state_dict(),
                           'best_loss': val_loss_G,
                           'epoch': epoch,
                           'optim': self.optimizer_G.state_dict(),
                           'lr_decay_G': self.lr_decay_G.state_dict()},
                          pt_name='netG')
                self.save({'net': self.netD.state_dict(),
                           'best_loss': val_loss_D,
                           'epoch': epoch,
                           'optim': self.optimizer_D.state_dict(),
                           'lr_decay_D': self.lr_decay_D.state_dict()},
                          pt_name='netD')
            if epoch == self.epochs - 1:
                logger.info('---Saving Latest---')
                self.save_lastet({'net': self.netG.state_dict(),
                                  'best_loss': val_loss_G,
                                  'epoch': epoch,
                                  'optim': self.optimizer_G.state_dict(),
                                  'lr_decay_G': self.lr_decay_G.state_dict()},
                                 pt_name='netG_latest')
                self.save_lastet({'net': self.netD.state_dict(),
                                  'best_loss': val_loss_D,
                                  'epoch': epoch,
                                  'optim': self.optimizer_D.state_dict(),
                                  'lr_decay_D_A': self.lr_decay_D.state_dict()},
                                 pt_name='netD_A_latest')
        logger.info(f"best val loss: {self.best_loss}")

        self.writer.close()

    def train(self, epoch):
        loss_G = AveMeter()
        loss_D = AveMeter()

        Tensor = torch.cuda.FloatTensor if self.config.cuda else torch.Tensor
        target_real = Variable(Tensor(1).fill_(1.0), requires_grad=False)  # real label 为1
        target_fake = Variable(Tensor(1).fill_(0.0), requires_grad=False)

        self.netG.train()
        self.netD.train()

        self.lr_decay_G.step()
        self.lr_decay_D.step()

        for i, (input_A, input_B) in enumerate(self.loaders['train']):

            if self.config.cuda:
                input_A = Variable(input_A).cuda()
                input_B = Variable(input_B).cuda()
            fake_B = self.netG(input_A)

            # (1) Update D network
            self.optimizer_D.zero_grad()

            # train with fake
            fake_ab = torch.cat((input_A, fake_B), 1)
            pred_fake = self.netD(fake_ab)  # fake_ab.detach()??
            loss_D_fake = self.criterion_GAN(pred_fake, target_fake)

            # train with real
            real_ab = torch.cat((input_A, input_B), 1)
            pred_real = self.netD(real_ab.detach())
            loss_D_real = self.criterion_GAN(pred_real, target_real)

            # combined D loss
            loss_D = (loss_D_fake + loss_D_real) * 0.5

            loss_D.backward()
            self.optimizer_D.step()

            # (2) Update G network
            self.optimizer_G.zero_grad()

            # G(A) should fake the D
            pred_fake = self.netD(fake_ab.detach())
            loss_G_l1 = self.criterion_GAN(pred_fake, target_real)
            # G(A) = B
            fake_B = self.netG(input_A.detach())
            loss_G_gan = self.criterion_L1(fake_B, input_B) * 10
            loss_G = loss_G_l1 + loss_G_gan

            loss_G.backward()
            self.optimizer_G.step()


            if i % 200 == 0 or i == len(self.loaders['train']) - 1:
                logger.info(f"Train: [{i}/{len(self.loaders['train'])}] | "
                            f"Time: {self.timer.timeSince()} | "
                            f"loss_G: {loss_G:.4f} | "
                            f"loss_D: {loss_D:.4f} | "
                            f"lr: {self.optimizer_G.param_groups[0]['lr']:.7f}"
                )

        self.writer.add_scalar('train/loss_G', loss_G, epoch)
        self.writer.add_scalar('train/loss_D', loss_D, epoch)
        self.writer.add_scalar('train/lr', self.optimizer_G.param_groups[0]['lr'], epoch)

    def val(self, epoch):
        loss_G = AveMeter()
        loss_D = AveMeter()

        Tensor = torch.cuda.FloatTensor if self.config.cuda else torch.Tensor
        target_real = Variable(Tensor(1).fill_(1.0), requires_grad=False)
        target_fake = Variable(Tensor(1).fill_(0.0), requires_grad=False)

        self.netG.eval()
        self.netD.eval()
        with torch.no_grad():
            for i, (input_A, input_B) in enumerate(self.loaders['train']):
                if self.config.cuda:
                    input_A = Variable(input_A).cuda()
                    input_B = Variable(input_B).cuda()
                fake_B = self.netG(input_A)

                # G_A & G_B #
                # (1) Update D network

                # train with fake
                fake_ab = torch.cat((input_A, fake_B), 1)
                pred_fake = self.netD(fake_ab)  # fake_ab.detach()??
                loss_D_fake = self.criterion_GAN(pred_fake, target_fake)

                # train with real
                real_ab = torch.cat((input_A, input_B), 1)
                pred_real = self.netD(real_ab)
                loss_D_real = self.criterion_GAN(pred_real, target_real)

                # combined D loss
                loss_D = (loss_D_fake + loss_D_real) * 0.5


                # (2) Update G network

                # G(A) should fake the D
                loss_G_l1 = self.criterion_GAN(pred_fake, target_real)
                # G(A) = B
                loss_G_gan = self.criterion_L1(pred_fake, input_B) * 10
                loss_G = loss_G_l1 + loss_G_gan


                # print('val D_A: {}, D_B: {}, A_size: {}\n'.format(loss_D_A, loss_D_B, input_A.size()[0]))

            logger.info(f"Val: [{i}/{len(self.loaders['val'])}] | "
                        f"Time: {self.timer.timeSince()} | "
                        f"loss_GAN_A2B: {loss_G:.4f} | "
                        f"loss_D: {loss_D:.4f}"
            )

        self.writer.add_scalar('val/loss_G', loss_G, epoch)
        self.writer.add_scalar('val/loss_D', loss_D, epoch)

        if epoch < 10 or epoch % 10 == 0:
            self.summary_imgs(input_A, input_B, self.netG(input_A), epoch)

        return loss_G, loss_D

    def lambda_rule(self, epoch):
        lr_l = 1.0 - max(0, epoch - self.lr_decay_start) / float(self.epochs + 1)
        return lr_l

    def save(self, state, pt_name):
        torch.save(state, os.path.join(self.ckpt_dir, '{}.pt'.format(pt_name)))
        logger.info('***Saving {} model***'.format(pt_name))
    
    def save_lastet(self, state, pt_name):
        torch.save(state, os.path.join(self.ckpt_dir, '{}.pt'.format(pt_name)))
        logger.info('***Saving {} model***'.format(pt_name))

    def save_config(self, config):
        with open(os.path.join(self.ckpt_dir, 'config.json'), 'w+') as f:
            f.write(json.dumps(config.__dict__, indent=4))
        f.close()

    def summary_imgs(self, imgs, targets, outputs, epoch):
        grid_imgs = make_grid(vis_seq(outputs[:3].cpu().data.numpy()),
                              nrow=4, normalize=True, range=(-1, 1))
        self.writer.add_image('fake_B', grid_imgs, epoch)
        grid_imgs = make_grid(vis_seq(targets[:3].cpu().data.numpy()),
                              nrow=4, normalize=True, range=(-1, 1))
        self.writer.add_image('real_B', grid_imgs, epoch)
        grid_imgs = make_grid(imgs[:4].clone().cpu().data, nrow=4, normalize=True, range=(-1, 1))
        self.writer.add_image('real_A', grid_imgs, epoch)
