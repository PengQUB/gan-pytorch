# -*- coding: utf-8 -*-
import json
import os
import logging
import torch
import itertools
from torch import optim, nn
from torch.autograd import Variable
from models import ModelSelector
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
        self.datasets, self.loaders = get_loader(config)
        self.max_iters = config.max_iters
        if self.max_iters is not None:
            self.epochs = self.max_iters // len(self.loaders['train'])
        else:
            self.epochs = config.epochs
        self.start_epoch = 0

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
            self.netG_A2B = self.netG_A2B.cuda()
            self.netG_B2A = self.netG_B2A.cuda()
            self.netD_A = self.netD_A.cuda()
            self.netD_B = self.netD_B.cuda()
            self.criterion_GAN = self.criterion_GAN.cuda()
            self.criterion_cycle = self.criterion_cycle.cuda()
            self.criterion_identity = self.criterion_identity.cuda()

        # self.criterion = LossSelector[config.loss](**config.loss_params[config.loss])
        self.optimizer_G = optim.Adam(itertools.chain(self.netG_A2B.parameters(),
                                                      self.netG_B2A.parameters()),
                                      lr=self.lr, betas=(0.9, 0.999), eps=1e-8)
        self.optimizer_D_A = optim.Adam(self.netD_A.parameters(), lr=self.lr, betas=(0.9, 0.999), eps=1e-8)
        self.optimizer_D_B = optim.Adam(self.netD_B.parameters(), lr=self.lr, betas=(0.9, 0.999), eps=1e-8)
        if self.max_iters is not None:
            self.lr_decay_G = optim.lr_scheduler.CosineAnnealingLR(self.optimizer_G, self.max_iters)
            self.lr_decay_D_A = optim.lr_scheduler.CosineAnnealingLR(self.optimizer_D_A, self.max_iters)
            self.lr_decay_D_B = optim.lr_scheduler.CosineAnnealingLR(self.optimizer_D_B, self.max_iters)
        elif self.epochs is not None:
            self.lr_decay_G = optim.lr_scheduler.CosineAnnealingLR(self.optimizer_G, self.epochs*len(self.loaders['train']))
            self.lr_decay_D_A = optim.lr_scheduler.CosineAnnealingLR(self.optimizer_D_A, self.epochs*len(self.loaders['train']))
            self.lr_decay_D_B = optim.lr_scheduler.CosineAnnealingLR(self.optimizer_D_B, self.epochs*len(self.loaders['train']))

        self.best_loss = float('inf')

        if config.resume:
            logger.info('***Resume from checkpoint***')
            state = torch.load(os.path.join(self.ckpt_dir, 'netG_A2B.pt'))
            self.netG_A2B.load_state_dict(state['net'])
            self.start_epoch = state['epoch']
            self.best_loss = state['best_loss']
            self.optimizer_G.load_state_dict(state['optim'])
            self.lr_decay_G.load_state_dict(state['lr_decay_G'])
            self.lr_decay_G.last_epoch = self.start_epoch

            state = torch.load(os.path.join(self.ckpt_dir, 'netG_B2A.pt'))
            self.netG_A2B.load_state_dict(state['net'])
            self.start_epoch = state['epoch']
            self.best_loss = state['best_loss']
            self.optimizer_G.load_state_dict(state['optim'])
            self.lr_decay_G.load_state_dict(state['lr_decay'])
            self.lr_decay_G.last_epoch = self.start_epoch

            state = torch.load(os.path.join(self.ckpt_dir, 'netD_A.pt'))
            self.netG_A2B.load_state_dict(state['net'])
            self.start_epoch = state['epoch']
            self.best_loss = state['best_loss']
            self.optimizer_D_A.load_state_dict(state['optim'])
            self.lr_decay_D_A.load_state_dict(state['lr_decay'])
            self.lr_decay_D_A.last_epoch = self.start_epoch

            state = torch.load(os.path.join(self.ckpt_dir, 'netD_B.pt'))
            self.netG_A2B.load_state_dict(state['net'])
            self.start_epoch = state['epoch']
            self.best_loss = state['best_loss']
            self.optimizer_D_B.load_state_dict(state['optim'])
            self.lr_decay_D_B.load_state_dict(state['lr_decay'])
            self.lr_decay_D_B.last_epoch = self.start_epoch

    def train_and_val(self):
        for epoch in range(self.start_epoch, self.epochs):
            logger.info(f"Epoch :[{epoch}/{self.epochs}]")
            self.train(epoch)
            logger.info(f"val starts...")
            val_loss_G, val_loss_G_identity, val_loss_G_GAN, val_loss_G_cycle, val_loss_D = self.val(epoch)
            if val_loss_G < self.best_loss:
                logger.info('------')
                self.best_loss = val_loss_G
                self.save({'net': self.netG_A2B.state_dict(),
                           'best_loss': val_loss_G,
                           'epoch': epoch,
                           'optim': self.optimizer_G.state_dict(),
                           'lr_decay': self.lr_decay_G.state_dict()},
                          pt_name='netG_A2B')
                self.save({'net': self.netG_B2A.state_dict(),
                           'best_loss': val_loss_G,
                           'epoch': epoch,
                           'optim': self.optimizer_G.state_dict(),
                           'lr_decay': self.lr_decay_G.state_dict()},
                          pt_name='netG_A2B')
                self.save({'net': self.netD_A.state_dict(),
                           'best_loss': val_loss_D,
                           'epoch': epoch,
                           'optim': self.optimizer_D_A.state_dict(),
                           'lr_decay': self.lr_decay_D_A.state_dict()},
                          pt_name='netG_A2B')
                self.save({'net': self.netD_B.state_dict(),
                           'best_loss': val_loss_D,
                           'epoch': epoch,
                           'optim': self.optimizer_D_B.state_dict(),
                           'lr_decay': self.lr_decay_D_B.state_dict()},
                          pt_name='netG_A2B')
        logger.info(f"best val loss: {self.best_loss}")

        self.writer.close()

    def train(self, epoch):
        losses_G = AveMeter()
        losses_G_identity = AveMeter()
        losses_G_GAN = AveMeter()
        losses_G_cycle = AveMeter()
        losses_D = AveMeter()

        Tensor = torch.cuda.FloatTensor if self.config.cuda else torch.Tensor
        target_real = Variable(Tensor(self.config.batch_size).fill_(1.0), requires_grad=False)
        target_fake = Variable(Tensor(self.config.batch_size).fill_(0.0), requires_grad=False)

        self.netG_A2B.train()
        self.netG_B2A.train()
        self.netD_A.train()
        self.netD_B.train()

        for i, (input_A, input_B) in enumerate(self.loaders['train']):
            self.lr_decay_G.step()
            self.lr_decay_D_A.step()
            self.lr_decay_D_B.step()

            if self.config.cuda:
                input_A = Variable(input_A).cuda()
                input_B = Variable(input_B).cuda()

            # G_A & G_B #
            ### identity_loss
            same_B = self.netG_A2B(input_B)
            loss_identity_B = self.criterion_identity(same_B, input_B) * 5.0

            same_A = self.netG_B2A(input_A)
            loss_identity_A = self.criterion_identity(same_A, input_A) * 5.0

            ### GAN_loss
            fake_B = self.netG_A2B(input_A)
            pred_fake = self.netD_B(fake_B)
            loss_GAN_A2B = self.criterion_GAN(pred_fake, target_real)

            fake_A = self.netG_B2A(input_B)
            pred_fake = self.netD_A(fake_A)
            loss_GAN_B2A = self.criterion_GAN(pred_fake, target_real)

            ### cycle_loss
            recovered_A = self.netG_B2A(fake_B)
            loss_cycle_ABA = self.criterion_cycle(recovered_A, input_A) * 10.0

            recovered_B = self.netG_A2B(fake_A)
            loss_cycle_BAB = self.criterion_cycle(recovered_B, input_B) * 10.0

            ### Total loss
            loss_G = loss_identity_A + loss_identity_B + loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB

            self.optimizer_G.zero_grad()
            loss_G.backward()
            self.optimizer_G.step()

            # D_A & D_B #
            fake_A_buffer = ReplayBuffer()
            fake_B_buffer = ReplayBuffer()

            ### D_A Real loss
            pred_real = self.netD_A(input_A)
            loss_D_real = self.criterion_GAN(pred_real, target_real)

            ### D_A Fake loss
            fake_A = fake_A_buffer.push_and_pop(fake_A)
            pred_fake = self.netD_A(fake_A.detach())
            loss_D_fake = self.criterion_GAN(pred_fake, target_fake)

            ### D_A Total loss
            loss_D_A = (loss_D_real + loss_D_fake) * 0.5

            self.optimizer_D_A.zero_grad()
            loss_D_A.backward()
            self.optimizer_D_A.step()

            ### D_B Real loss
            pred_real = self.netD_B(input_B)
            loss_D_real = self.criterion_GAN(pred_real, target_real)

            ### D_B Fake loss
            fake_B = fake_B_buffer.push_and_pop(fake_B)
            pred_fake = self.netD_B(fake_B.detach())
            loss_D_fake = self.criterion_GAN(pred_fake, target_fake)

            ### D_B Total loss
            loss_D_B = (loss_D_real + loss_D_fake) * 0.5

            self.optimizer_D_B.zero_grad()
            loss_D_B.backward()
            self.optimizer_D_B.step()

            losses_G.update(loss_G.item(), input_A.size()[0])
            losses_G_identity.update((loss_identity_A + loss_identity_B).item(), input_A.size()[0])
            losses_G_GAN.update((loss_GAN_A2B + loss_GAN_B2A).item(), input_A.size()[0])
            losses_G_cycle.update((loss_cycle_ABA + loss_cycle_BAB).item(), input_A.size()[0])
            losses_D.update((loss_D_A + loss_D_B).item(), input_A.size()[0])

            if i % 200 == 0 or i == len(self.loaders['train']) - 1:
                logger.info(f"Train: [{i}/{len(self.loaders['train'])}] | "
                            f"Time: {self.timer.timeSince()} | "
                            f"loss_G: {losses_G.avg:.4f} | "
                            f"loss_G_identity:{losses_G_identity.avg:.4f} | "
                            f"loss_G_GAN: {losses_G_GAN.avg:.4f} | "
                            f"loss_G_cycle: {losses_G_cycle.avg:.4f} | "
                            f"loss_D: {losses_D.avg:.4f}"
                )

        self.writer.add_scalar('train/loss_G', losses_G.avg, epoch)
        self.writer.add_scalar('train/loss_G_identity', losses_G_identity.avg, epoch)
        self.writer.add_scalar('train/loss_G_GAN', losses_G_GAN.avg, epoch)
        self.writer.add_scalar('train/loss_G_cycle', losses_G_cycle.avg, epoch)
        self.writer.add_scalar('train/loss_D', losses_D.avg, epoch)

    def val(self, epoch):
        losses_G = AveMeter()
        losses_G_identity = AveMeter()
        losses_G_GAN = AveMeter()
        losses_G_cycle = AveMeter()
        losses_D = AveMeter()

        Tensor = torch.cuda.FloatTensor if self.config.cuda else torch.Tensor
        target_real = Variable(Tensor(self.config.batch_size).fill_(1.0), requires_grad=False)
        target_fake = Variable(Tensor(self.config.batch_size).fill_(0.0), requires_grad=False)

        self.netG_A2B.eval()
        self.netG_B2A.eval()
        self.netD_A.eval()
        self.netD_B.eval()
        with torch.no_grad():
            for i, (input_A, input_B) in enumerate(self.loaders['val']):
                if self.config.cuda:
                    input_A = Variable(input_A).cuda()
                    input_B = Variable(input_B).cuda()

                # G_A & G_B #
                ### identity_loss
                same_B = self.netG_A2B(input_B)
                loss_identity_B = self.criterion_identity(same_B, input_B) * 5.0

                same_A = self.netG_B2A(input_A)
                loss_identity_A = self.criterion_identity(same_A, input_A) * 5.0

                ### GAN_loss
                fake_B = self.netG_A2B(input_A)
                pred_fake = self.netD_B(fake_B)
                loss_GAN_A2B = self.criterion_GAN(pred_fake, target_real)

                fake_A = self.netG_B2A(input_B)
                pred_fake = self.netD_A(fake_A)
                loss_GAN_B2A = self.criterion_GAN(pred_fake, target_real)

                ### cycle_loss
                recovered_A = self.netG_B2A(fake_B)
                loss_cycle_ABA = self.criterion_cycle(recovered_A, input_A) * 10.0

                recovered_B = self.netG_A2B(fake_A)
                loss_cycle_BAB = self.criterion_cycle(recovered_B, input_B) * 10.0

                ### Total loss
                loss_G = loss_identity_A + loss_identity_B + loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB


                # D_A & D_B #
                fake_A_buffer = ReplayBuffer()
                fake_B_buffer = ReplayBuffer()

                ### D_A Real loss
                pred_real = self.netD_A(input_A)
                loss_D_real = self.criterion_GAN(pred_real, target_real)

                ### D_A Fake loss
                fake_A = fake_A_buffer.push_and_pop(fake_A)
                pred_fake = self.netD_A(fake_A.detach())
                loss_D_fake = self.criterion_GAN(pred_fake, target_fake)

                ### D_A Total loss
                loss_D_A = (loss_D_real + loss_D_fake) * 0.5

                ### D_B Real loss
                pred_real = self.netD_B(input_B)
                loss_D_real = self.criterion_GAN(pred_real, target_real)

                ### D_B Fake loss
                fake_B = fake_B_buffer.push_and_pop(fake_B)
                pred_fake = self.netD_B(fake_B.detach())
                loss_D_fake = self.criterion_GAN(pred_fake, target_fake)

                ### D_B Total loss
                loss_D_B = (loss_D_real + loss_D_fake) * 0.5

                losses_G.update(loss_G.item(), input_A.size()[0])
                losses_G_identity.update((loss_identity_A + loss_identity_B).item(), input_A.size()[0])
                losses_G_GAN.update((loss_GAN_A2B + loss_GAN_B2A).item(), input_A.size()[0])
                losses_G_cycle.update((loss_cycle_ABA + loss_cycle_BAB).item(), input_A.size()[0])
                losses_D.update((loss_D_A + loss_D_B).item(), input_A.size()[0])

            logger.info(f"Val: [{i}/{len(self.loaders['val'])}] | "
                        f"Time: {self.timer.timeSince()} | "
                        f"loss_G: {losses_G.avg:.4f} | "
                        f"loss_G_identity:{losses_G_identity.avg:.4f} | "
                        f"loss_G_GAN: {losses_G_GAN.avg:.4f} | "
                        f"loss_G_cycle: {losses_G_cycle.avg:.4f} | "
                        f"loss_D: {losses_D.avg:.4f}"
            )

        self.writer.add_scalar('val/loss_G', losses_G.avg, epoch)
        self.writer.add_scalar('val/loss_G_identity', losses_G_identity.avg, epoch)
        self.writer.add_scalar('val/loss_G_GAN', losses_G_GAN.avg, epoch)
        self.writer.add_scalar('val/loss_G_cycle', losses_G_cycle.avg, epoch)
        self.writer.add_scalar('val/loss_D', losses_D.avg, epoch)

        if epoch % 10 == 0:
            self.summary_imgs(input_A, input_B, fake_B, epoch)

        return losses_G.avg, losses_G_identity.avg, losses_G_GAN.avg, losses_G_cycle.avg, losses_D.avg

    def save(self, state, pt_name):
        torch.save(state, os.path.join(self.ckpt_dir, '{}.pt'.format(pt_name)))
        logger.info('***Saving model***')

    def save_config(self, config):
        with open(os.path.join(self.ckpt_dir, 'config.json'), 'w+') as f:
            f.write(json.dumps(config.__dict__, indent=4))
        f.close()

    def summary_imgs(self, imgs, targets, outputs, epoch):
        grid_imgs = make_grid(imgs[:3].clone().cpu().data, nrow=3, normalize=True)
        self.writer.add_image('real_A', grid_imgs, epoch)
        grid_imgs = make_grid(vis_seq(outputs[:3].cpu().data.numpy()),
                              nrow=3, normalize=False, range=(0, 255))
        self.writer.add_image('real_B', grid_imgs, epoch)
        grid_imgs = make_grid(vis_seq(targets[:3].cpu().data.numpy()),
                              nrow=3, normalize=False, range=(0, 255))
        self.writer.add_image('fake_B', grid_imgs, epoch)
