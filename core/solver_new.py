import os
import numpy as np
import time
import datetime
import torch
import torchvision
from torch import optim
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn

from core.evaluation import *
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, median_absolute_error, precision_score, recall_score, accuracy_score, f1_score
import csv
from torchinfo import summary
import wandb
import matplotlib.pyplot as plt  # plotting tools
import torchvision.models as models
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
from utils.wandb_utils import Wandb
from utils.model_checkpoint import ModelCheckpoint
from scipy.optimize import curve_fit
from scipy.stats import linregress
from functools import partial
from torch import Tensor
import ipdb




def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False



class Solver(object):
    def __init__(self, config, train_loader, valid_loader, test_loader, patch_size, launch_wandb=1):
        torch.cuda.empty_cache()

        # Data loader
        self.config = config
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.year = config.year
        self.patch_size = patch_size
        self.conf_score = config.conf_score
        self.norm_wei = config.norm_wei
        # Models
        self.nnet = None
        self.pretrained = config.pretrained
        self.keep_all_features = config.keep_all_features
        self.optimizer = None
        self.img_ch = config.img_ch
        self.output_ch = config.output_ch


        self.loss_name = config.loss_name
        self.augmentation_prob = config.augmentation_prob
        self.nnlevel = config.nnlevel
        # Hyper-parameters
        self.lr = config.lr
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.weightDecay = config.weightDecay

        # Training settings
        self.num_epochs = config.num_epochs
        # self.num_epochs_decay = config.num_epochs_decay
        self.batch_size = config.batch_size
        self.suf = config.model_suf
        self.lr_decay_rate = config.lr_decay_rate
        self.lr_decay_frequency = config.lr_decay_frequency
        self.min_lr = config.min_lr
        self.add_input2 = config.add_input2
        self.add_outputs = config.add_outputs

        reduc = 'mean'

        self.loss_func = config.loss_func
        if self.loss_func == 'L1':
            print('Loss function: L1')
            self.criterion = torch.nn.L1Loss(reduction=reduc)
        elif self.loss_func == 'L2':
            # try log loss
            print('Loss function: L2')
            self.criterion = torch.nn.MSELoss(reduction=reduc)
        elif self.loss_func == 'BCE':
            self.criterion = torch.nn.BCELoss(reduction=reduc)
        elif self.loss_func == 'focal':
            self.criterion = Loss(loss_type="focal_loss")
        elif self.loss_func == 'BCE2':
            self.criterion = Loss(loss_type="binary_cross_entropy")
        elif self.loss_func == 'focal_balance':
            self.criterion = Loss(loss_type="focal_loss", samples_per_class=self.config.sample_counts, class_balanced=True)

        else:
            print('loss not specified')

        if self.add_outputs:
            self.loss_func_addop = config.loss_func_addop
            print('Loss function for added outputs: ', self.loss_func_addop)
            if self.loss_func_addop == 'L1':
                self.criterion_addop = torch.nn.L1Loss(reduction=reduc)
            elif self.loss_func_addop == 'L2':
                self.criterion_addop = torch.nn.MSELoss(reduction=reduc)
            elif self.loss_func_addop == 'BCElog':
                self.criterion_addop = torch.nn.BCEWithLogitsLoss(reduction=reduc)

        # Step size
        self.log_step = config.log_step
        self.val_step = config.val_step

        # Path
        self.model_path = config.model_path
        self.result_path = config.result_path
        self.mode = config.mode
        self.checkpoint_dir = config.checkpoint_dir
        self.image_callback_freq = config.image_callback_freq


        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_type = config.model_type
        self.t = config.t
        self.build_model()



        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        timestr = time.strftime("%Y%m%d-%H%M")
        if self.add_input2:
            ps = self.patch_size[0][-1]
        else:
            ps = self.patch_size[-1]

        if config.onlyRGB:
            if not config.add_chm:
                self.bandns = 'RGB'
            elif config.add_chm:
                self.bandns = 'RGBCHM'
        else:
            print('invalid band setting: check!')

        if config.task == 'classification':
            self.nnet_path_dir = os.path.join(self.model_path, 'classification_%s-%s-Epo%d-patch_%d-Loss_%s-bands_%s-_%s' %(timestr, self.model_type,self.num_epochs, ps, self.loss_func, self.bandns, self.suf))

        print('********************************************')
        print('Model path: ', self.nnet_path_dir)
        if self.mode == 'train' and not os.path.exists(self.nnet_path_dir):
            os.makedirs(self.nnet_path_dir)
# =============================================================================
        self.saveImages = config.saveImages
        if self.saveImages:
            self.image_callback_dir = os.path.join(config.imageCallbackDir, 'model_%s-%s'%(timestr, self.model_type))
            if not os.path.exists(self.image_callback_dir):
                os.makedirs(self.image_callback_dir)

        # adding wandb

        self._checkpoint: ModelCheckpoint = ModelCheckpoint(
        self.checkpoint_dir,
        self.config.model_type,
        self.config.mode,
        run_config=self.config,
        resume=0,
        )

        if self.mode == 'train' and launch_wandb:
            Wandb.launch(config, 1)
# =============================================================================





    def build_model(self):
        """Build generator and discriminator."""
        if 'torchEfficientnetb0' in self.model_type:
            if self.config.task == 'classification':
                # use default arch
                self.nnet = models.efficientnet_b0(pretrained=self.pretrained)
                # ipdb.set_trace()
                num_fs = self.nnet.classifier[1].in_features
                self.nnet.classifier = torch.nn.Sequential(nn.Dropout(p=0.2, inplace = True),
                                                           nn.Linear(in_features=num_fs, out_features=640, bias=True),
                                                           nn.Dropout(p=0.2, inplace = True),
                                                           nn.Linear(in_features=640, out_features=320, bias=True),
                                                           nn.Dropout(p=0.2, inplace = True),
                                                           nn.Linear(in_features=320, out_features=64, bias=True),
                                                           nn.Dropout(p=0.2, inplace = True),
                                                           nn.Linear(in_features=64, out_features=1, bias=True))
            else:
                raise NotImplementedError

        elif 'torchEfficientnetb1' in self.model_type:

            if self.config.task == 'classification':
                self.nnet = models.efficientnet_b1(pretrained=self.pretrained)
                num_fs = self.nnet.classifier[1].in_features
                self.nnet.classifier = torch.nn.Sequential(nn.Dropout(p=0.2, inplace = True),
                                                           nn.Linear(in_features=num_fs, out_features=640, bias=True),
                                                           nn.Dropout(p=0.2, inplace = True),
                                                           nn.Linear(in_features=640, out_features=320, bias=True),
                                                           nn.Dropout(p=0.2, inplace = True),
                                                           nn.Linear(in_features=320, out_features=64, bias=True),
                                                           nn.Dropout(p=0.2, inplace = True),
                                                           nn.Linear(in_features=64, out_features=1, bias=True))

            else:
                raise NotImplementedError

        elif 'torchEfficientnetb2' in self.model_type:
            if self.config.task == 'classification':
                self.nnet = models.efficientnet_b2(pretrained=self.pretrained)
                num_fs = self.nnet.classifier[1].in_features
                self.nnet.classifier = torch.nn.Sequential(nn.Dropout(p=0.2, inplace = True),
                                                           nn.Linear(in_features=num_fs, out_features=640, bias=True),
                                                           nn.Dropout(p=0.2, inplace = True),
                                                           nn.Linear(in_features=640, out_features=320, bias=True),
                                                           nn.Dropout(p=0.2, inplace = True),
                                                           nn.Linear(in_features=320, out_features=64, bias=True),
                                                           nn.Dropout(p=0.2, inplace = True),
                                                           nn.Linear(in_features=64, out_features=1, bias=True))

            else:
                raise NotImplementedError

        elif self.model_type == 'resnet18':
            self.nnet = models.resnet18(pretrained=self.pretrained, num_classes=self.output_ch)
            self.nnet.conv1 = nn.Conv2d(3, 64, (7, 7), (1, 1), (3, 3), bias=False)

        else:
            raise NotImplementedError

        if not self.pretrained:

            self.optimizer = optim.Adam(list(self.nnet.parameters()),
                                      self.lr, [self.beta1, self.beta2], self.weightDecay)

        else: # using pretrained model
            params_to_update = self.nnet.parameters()
            print("Params to learn:")

            params_to_update = []
            for name,param in self.nnet.named_parameters():
                if param.requires_grad == True:
                    params_to_update.append(param)
                    print("\t",name)


            # Observe that all parameters are being optimized
            self.optimizer = optim.Adam(params_to_update, self.lr, [self.beta1, self.beta2], self.weightDecay)



        self.nnet.to(self.device)

        self.print_network(self.nnet, self.model_type)



    def print_network(self, model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        # print(model)
        # import ipdb
        # ipdb.set_trace()

        print('------Model name------- ', name)
        if self.add_input2:
            model_stats = summary(model, input_size=[self.patch_size[0], self.patch_size[1]], col_names = ("input_size", "output_size", "num_params"))
        else:
            model_stats = summary(model, input_size=self.patch_size, col_names = ("input_size", "output_size", "num_params"))
        # model_stats = summary(model, input_size=(32, 3, 180, 180))
        print(model_stats)
        print("The number of parameters: {}".format(num_params))
        self.summary_str = str(model_stats)





    def train_epoch(self, epoch, lr):
        self.nnet.train(True)
        self.stage = 'train'
        batch_time = AverageMeter()
        losses = AverageMeter()
        if self.config.task == 'classification':
            f1 = AverageMeter()
            acc = AverageMeter()
            sens = AverageMeter()
            prec = AverageMeter()
        else:
            raise NotImplementedError
        if self.add_outputs:
            loss_ad1 = AverageMeter()
            loss_ad2 = AverageMeter()
            loss_ad3 = AverageMeter()


        end = time.time()
        st = time.time()
        for i, data in enumerate(self.train_loader):

            if self.conf_score:
                if self.add_input2:
                    images, input2, GT, sp_wei = data
                elif self.add_outputs:
                    images, GT, add_op1, add_op2, sp_wei = data
                else:
                    images, GT, sp_wei = data
            else:
                images, GT = data
            # GT : Ground Truth

            images = images.to(self.device)
            GT = GT.to(self.device)

            if self.conf_score:
                sp_wei = sp_wei.to(self.device)
            if self.add_input2:
                input2 = input2.to(self.device)
            if self.add_outputs:
                add_op1, add_op2 = add_op1.to(self.device), add_op2.to(self.device)

            if self.add_input2:
                pred = self.nnet(images, input2).squeeze()
            else:
                if self.add_outputs:
                    pred = self.nnet(images)
                else:
                    pred = self.nnet(images).squeeze()

            if self.add_outputs:
                pred1, pred2, pred3 = pred
                pred1, pred2, pred3 = pred1.squeeze(), pred2.squeeze(), pred3.squeeze()

                loss1 = self.criterion(pred1,GT) # AGB loss
                loss2 = self.criterion_addop(pred2, add_op1)
                loss3 = self.criterion_addop(pred3, add_op2)
                loss = loss1 + loss2 + loss3 # weighted loss for 3 outputs
            else:   # one output
                if 'map' in self.model_type:
                    # need to sum first
                    pred = torch.sum(pred, (-1))

                if self.config.task == 'classification':
                    if self.config.loss_func == 'BCE':
                        pred = torch.sigmoid(pred)
                GT = GT.to(torch.float32)
                loss = self.criterion(pred,GT)

            if self.conf_score:

                if self.norm_wei:

                    loss =(loss * sp_wei).sum() / sp_wei.sum()

                    if self.add_outputs:
                        loss1 =(loss1 * sp_wei).sum() / sp_wei.sum()
                        loss2 =(loss2 * sp_wei).sum() / sp_wei.sum()
                        loss3 =(loss3 * sp_wei).sum() / sp_wei.sum()

                else:
                    loss = torch.mean(loss*sp_wei)

            if self.conf_score:
                losses.update(loss.item(), sp_wei.sum().item())
            else:
                losses.update(loss.item(), images.size(0))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if self.add_outputs:
                pred1 = pred1.cpu().detach().numpy()
                pred2 = pred2.cpu().detach().numpy()
                pred3 = pred3.cpu().detach().numpy()
                add_op1 = add_op1.cpu().detach().numpy()
                add_op2 = add_op2.cpu().detach().numpy()
            else:
                pred = pred.cpu().detach().numpy()
            GT = GT.cpu().detach().numpy()
            if self.conf_score:
                sp_wei = sp_wei.cpu().detach().numpy().mean() #check if mean or not
                # import ipdb
                # ipdb.set_trace()

            if self.add_outputs:
                pred = pred1 # loss is the combined loss, but other metrics are only for AGB


            if self.config.task == 'classification':
                # ipdb.set_trace()
                pred_lb = pred>0.5
                pred_lb = pred_lb.astype('int8')
                GT = GT.astype('int8')
                f1.update(f1_score(GT, pred_lb).item(), images.size(0))
                acc.update(accuracy_score(GT, pred_lb).item(), images.size(0))
                sens.update(recall_score(GT, pred_lb).item(), images.size(0))
                prec.update(precision_score(GT, pred_lb).item(), images.size(0))

            if self.add_outputs:
                # import ipdb
                # ipdb.set_trace()
                loss_ad1.update(loss1.item(), sp_wei.sum().item())
                loss_ad2.update(loss2.item(), sp_wei.sum().item())
                loss_ad3.update(loss3.item(), sp_wei.sum().item())

            batch_time.update(time.time() - end)
            end = time.time()

            # save image callback
            if self.saveImages:
                if i == 0 and epoch % self.image_callback_freq == 0:
                    images = images.cpu().detach().numpy()
                    self.image_callback(epoch, images, pred, GT, maes.avg)



        if self.config.task == 'classification':
            print('Epoch [%d/%d] Epo time [%.2f] Itr time [%.2f] - Training, Loss: %.4f, acc: %.4f, F1: %.4f, Sens: %.4f, Prec: %.4f' %
                  (epoch+1, self.num_epochs, time.time() - st, batch_time.avg, \
                  losses.avg, acc.avg,\
                  f1.avg, sens.avg ,prec.avg))
        else:
            raise NotImplementedError



        # Decay learning rate

        if epoch % self.lr_decay_frequency == 0:

            lr = self.lr * (self.lr_decay_rate ** (epoch // self.lr_decay_frequency))
            if lr > self.min_lr:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr
                print('Decay learning rate to {}.'.format(lr))


        if self.config.task == 'classification':
            self.finalize_epoch_classification(epoch, losses.avg, f1.avg, sens.avg)
        else:
            raise NotImplementedError

        del images, GT
        if self.config.task == 'classification':
            return losses.avg, f1.avg, sens.avg



    def valid_epoch(self, epoch):
        st = time.time()
        self.nnet.train(False)
        self.nnet.eval()
        self.stage = 'val'
        batch_time = AverageMeter()
        val_losses = AverageMeter()

        if self.config.task == 'classification':
            val_f1 = AverageMeter()
            val_acc = AverageMeter()
            val_sens = AverageMeter()
            val_prec = AverageMeter()

        else:
            raise NotImplementedError

        if self.add_outputs:
            loss_ad1 = AverageMeter()
            loss_ad2 = AverageMeter()
            loss_ad3 = AverageMeter()
        val_r2 = AverageMeter()
        end = time.time()

        for i, data in enumerate(self.valid_loader):

            if self.conf_score:
                if self.add_input2:
                    images, input2, GT, sp_wei = data
                elif self.add_outputs:
                    images, GT, add_op1, add_op2, sp_wei = data
                else:
                    images, GT, sp_wei = data

            else:
                images, GT = data
            images = images.to(self.device)
            GT = GT.to(self.device)

            if self.conf_score:
                sp_wei = sp_wei.to(self.device)

            if self.add_input2:
                input2 = input2.to(self.device)

            if self.add_outputs:
                add_op1, add_op2 = add_op1.to(self.device), add_op2.to(self.device)


            if self.add_input2:
                pred = self.nnet(images, input2).squeeze()
            else:
                if self.add_outputs:
                    pred = self.nnet(images)
                else:
                    pred = self.nnet(images).squeeze()



            if self.add_outputs:
                pred1, pred2, pred3 = pred
                pred1, pred2, pred3 = pred1.squeeze(), pred2.squeeze(), pred3.squeeze()
                loss1 = self.criterion(pred1,GT) # AGB loss
                loss2 = self.criterion_addop(pred2, add_op1)
                loss3 = self.criterion_addop(pred3, add_op2)
                loss = loss1 + loss2 + loss3 # weighted loss for 3 outputs
            else:   # one output
                if 'map' in self.model_type:
                    # need to sum first
                    pred = torch.sum(pred, (-1))

                if self.config.task == 'classification':
                    if self.config.loss_func == 'BCE':
                        pred = torch.sigmoid(pred)

                GT = GT.to(torch.float32)
                loss = self.criterion(pred,GT)
            # adding sample weights to the loss
            # if True: loss(reduction = 'none)
            if self.conf_score:
                if self.norm_wei:
                    # # normlize loss weights
                    # loss =(loss * sp_wei / sp_wei.sum()).sum()
                    # loss = loss.mean()
                    # do not take mean
                    loss =(loss * sp_wei).sum() / sp_wei.sum()
                    if self.add_outputs:
                        loss1 =(loss1 * sp_wei).sum() / sp_wei.sum()
                        loss2 =(loss2 * sp_wei).sum() / sp_wei.sum()
                        loss3 =(loss3 * sp_wei).sum() / sp_wei.sum()

                else:
                    loss = torch.mean(loss*sp_wei)

            # val_losses.update(loss.item(), images.size(0))
            # val_losses.update(loss.item(), torch.count_nonzero(sp_wei).item())
            if self.conf_score:
                val_losses.update(loss.item(), sp_wei.sum().item())
            else:
                val_losses.update(loss.item(), images.size(0))
            # Calculate Metrics #
            if self.add_outputs:
                pred1 = pred1.cpu().detach().numpy()
                pred2 = pred2.cpu().detach().numpy()
                pred3 = pred3.cpu().detach().numpy()
                add_op1 = add_op1.cpu().detach().numpy()
                add_op2 = add_op2.cpu().detach().numpy()
            else:
                pred = pred.cpu().detach().numpy()

            GT = GT.cpu().detach().numpy()
            if self.conf_score:
                sp_wei = sp_wei.cpu().detach().numpy().mean() #check if mean or not
            if self.add_outputs:
                pred = pred1 # loss is the combined loss, but other metrics are only for AGB
            if self.config.task == 'classification':
                pred_lb = pred>0.5
                pred_lb = pred_lb.astype('int8')
                GT = GT.astype('int8')
                val_f1.update(f1_score(GT, pred_lb).item(), images.size(0))
                val_acc.update(accuracy_score(GT, pred_lb).item(), images.size(0))
                val_sens.update(recall_score(GT, pred_lb).item(), images.size(0))
                val_prec.update(precision_score(GT, pred_lb).item(), images.size(0))
            if self.add_outputs:
                loss_ad1.update(loss1.item(), sp_wei.sum().item())
                loss_ad2.update(loss2.item(), sp_wei.sum().item())
                loss_ad3.update(loss3.item(), sp_wei.sum().item())

            batch_time.update(time.time() - end)
            end = time.time()

            # save image callback
            if self.saveImages:
                if i == 0 and epoch % self.image_callback_freq == 0:
                    images = images.cpu().detach().numpy()
                    self.image_callback(epoch, images, pred, GT, val_maes.avg, stage = '_valid')

        if self.config.task == 'classification':
            print('Epoch [%d/%d] Epo time [%.2f] Itr time [%.2f] - Validation, Loss: %.4f, acc: %.4f, F1: %.4f, Sens: %.4f, Prec: %.4f' %
                  (epoch+1, self.num_epochs, time.time() - st, batch_time.avg, \
                  val_losses.avg, val_acc.avg,\
                  val_f1.avg, val_sens.avg ,val_prec.avg))

        if self.config.task == 'classification':
            self.finalize_epoch_classification(epoch, val_losses.avg, val_f1.avg, val_sens.avg)

        del images, GT

        if self.config.task == 'classification':
            return val_losses.avg, val_f1.avg, val_sens.avg



    def image_callback(self, epoch, img, pred, lab, mae, stage = '_train'):
        plt.figure(figsize = (20,20)) # size(width, height)
        for i in range(4):
            for j in range(4): # column
                plt.subplot(4, 4, 4*j + i+1)
                curim = img[4*j + i]
                curim = np.transpose(curim, axes=(1,2,0)) # Channel at the end

                # recover from meanstd
                imstd = np.array([0.229, 0.224, 0.225])
                immean = np.array([0.485, 0.456, 0.406])
                curim = curim*imstd + immean
                plt.imshow(curim[:, :, :3])
                plt.title("label: %.1f;\n pred: %.1f"%(lab[4*j + i], pred[4*j + i]))

        plt.tight_layout()
        figpath = os.path.join(self.image_callback_dir, 'Epoch' + str(epoch) + stage + '_MAE_' + str(round(mae)) + '.jpg')
        plt.savefig(figpath, quality = 30)
        plt.clf()
        plt.close('all')






    def finalize_epoch_classification(self, epoch, losses, f1, sens):
        met_names = ['loss', 'f1', 'sens']
        values = [losses, f1, sens]

        metr = {}
        for k in range(len(met_names)):
            metr[met_names[k]]=values[k]

        metr2 = get_metrics_classification(metr, self.stage)

        metrics = {}
        metrics['epoch']=epoch
        metrics['stage']=self.stage

        metrics['current_metrics'] = metr2
        # import ipdb
        # ipdb.set_trace()
        wandb.log(metr2, step=epoch)


        # self.publish_to_tensorboard(metrics, step)


        self._checkpoint.save_best_models_under_current_metrics(self.nnet, metrics)
        Wandb.add_file(self._checkpoint.checkpoint_path)

        wandb.config.update({"model_name": self.config.model_type}, allow_val_change=True)


    def train(self):
        #====================================== Training ===========================================#
        #===========================================================================================#
        print('--------------------------------------------------------------------------------------')
        nnet_path_model = os.path.join(self.nnet_path_dir, 'bestLoss.pkl')
        # U-Net Train
        if os.path.isfile(nnet_path_model):
            # Load the pretrained Encoder
            self.nnet.load_state_dict(torch.load(nnet_path_model))
            print('%s Weights are Successfully Loaded from %s'%(self.model_type,nnet_path_model))

            # read saved model architecture file
            print('Model architecture loaded from %s'%(nnet_path_model.replace('pkl', 'log')))
            with open(nnet_path_model.replace('pkl', 'log'), 'r') as f:
                lines = f.read()
                print(lines)
        else:
            # train other models
            lr = self.lr
            best_score = 100000
            if self.config.task == 'classification':
                best_f1 = -100
                best_sens = -100
            if self.config.earlystop:
                early_stopper = EarlyStopper(patience=self.config.patience)
            for epoch in range(self.num_epochs):

                print('-++++++++++++++++++++++------------------------------------------------------------------------------')
                losses, met1, met2 = self.train_epoch(epoch, lr)

                val_loss, val_met1, val_met2 = self.valid_epoch(epoch)

                if self.config.earlystop:
                    if early_stopper.early_stop(val_loss):
                        break

                # Save Best model
                if val_loss < best_score:
                    best_score = val_loss
                    best_nnet = self.nnet.state_dict()
                    print('==========================================================')
                    print('Best %s model score: Loss = %.4f'%(self.model_type,best_score))
                    torch.save(best_nnet,nnet_path_model)
                    if not os.path.isfile(nnet_path_model.replace('pkl', 'log')):
                        with open(nnet_path_model.replace('pkl', 'log'), 'w') as f:
                            f.write(self.summary_str)

                if self.config.task == 'classification':
                    if val_met1 > best_f1: # met1: f1 score
                        best_f1 = val_met1
                        best_nnet = self.nnet.state_dict()
                        print('==========================================================')
                        print('Best %s model score: F1 = %.4f'%(self.model_type,best_f1))
                        torch.save(best_nnet,nnet_path_model.replace('Loss.pkl', 'F1.pkl'))




    def test(self):
        #===================================== Test ====================================#
        # after all training epochs
        # self.build_model()
        if os.path.isfile(self.model_path):
            # Load the pretrained Encoder
            self.nnet.load_state_dict(torch.load(self.model_path))
            print('%s is Successfully Loaded from %s'%(self.model_type,self.model_path))
        # self.unet.load_state_dict(torch.load(self.model_path))
        self.nnet.train(False)
        self.nnet.eval()

        ####################################################3
        batch_time = AverageMeter()
        val_losses = AverageMeter()

        if self.config.task == 'classification':
            val_f1 = AverageMeter()
            val_acc = AverageMeter()
            val_sens = AverageMeter()
            val_prec = AverageMeter()

        if self.add_outputs:
            loss_ad1 = AverageMeter()
            loss_ad2 = AverageMeter()
            loss_ad3 = AverageMeter()
        val_r2 = AverageMeter()
        end = time.time()

        lb_list = []
        pd_list = []

        for i, data in enumerate(self.test_loader):

            if self.conf_score:
                if self.add_input2:
                    images, input2, GT, sp_wei = data
                elif self.add_outputs:
                    images, GT, add_op1, add_op2, sp_wei = data
                else:
                    images, GT, sp_wei = data

            else:
                images, GT = data
            images = images.to(self.device)
            GT = GT.to(self.device)

            if self.conf_score:
                sp_wei = sp_wei.to(self.device)

            if self.add_input2:
                input2 = input2.to(self.device)

            if self.add_outputs:
                add_op1, add_op2 = add_op1.to(self.device), add_op2.to(self.device)


            if self.add_input2:
                pred = self.nnet(images, input2).squeeze()
            else:
                if self.add_outputs:
                    pred = self.nnet(images)
                else:
                    pred = self.nnet(images).squeeze()



            if self.add_outputs:
                pred1, pred2, pred3 = pred
                pred1, pred2, pred3 = pred1.squeeze(), pred2.squeeze(), pred3.squeeze()
                loss1 = self.criterion(pred1,GT) # AGB loss
                loss2 = self.criterion_addop(pred2, add_op1)
                loss3 = self.criterion_addop(pred3, add_op2)
                loss = loss1 + loss2 + loss3 # weighted loss for 3 outputs
            else:   # one output
                if 'map' in self.model_type:
                    # need to sum first
                    pred = torch.sum(pred, (-1))

                if self.config.task == 'classification':

                    pred = torch.sigmoid(pred)

                GT = GT.to(torch.float32)
                loss = self.criterion(pred,GT)
            # adding sample weights to the loss
            # if True: loss(reduction = 'none)
            if self.conf_score:
                if self.norm_wei:
                    # # normlize loss weights
                    # loss =(loss * sp_wei / sp_wei.sum()).sum()
                    # loss = loss.mean()
                    # do not take mean
                    loss =(loss * sp_wei).sum() / sp_wei.sum()
                    if self.add_outputs:
                        loss1 =(loss1 * sp_wei).sum() / sp_wei.sum()
                        loss2 =(loss2 * sp_wei).sum() / sp_wei.sum()
                        loss3 =(loss3 * sp_wei).sum() / sp_wei.sum()

                else:
                    loss = torch.mean(loss*sp_wei)

            # val_losses.update(loss.item(), images.size(0))
            # val_losses.update(loss.item(), torch.count_nonzero(sp_wei).item())
            if self.conf_score:
                val_losses.update(loss.item(), sp_wei.sum().item())
            else:
                val_losses.update(loss.item(), images.size(0))
            # Calculate Metrics #
            if self.add_outputs:
                pred1 = pred1.cpu().detach().numpy()
                pred2 = pred2.cpu().detach().numpy()
                pred3 = pred3.cpu().detach().numpy()
                add_op1 = add_op1.cpu().detach().numpy()
                add_op2 = add_op2.cpu().detach().numpy()
            else:
                pred = pred.cpu().detach().numpy()

            GT = GT.cpu().detach().numpy()
            if self.conf_score:
                sp_wei = sp_wei.cpu().detach().numpy().mean() #check if mean or not
            if self.add_outputs:
                pred = pred1 # loss is the combined loss, but other metrics are only for AGB
            if self.config.task == 'classification':
                pred_lb = pred>0.5
                pred_lb = pred_lb.astype('int8')
                GT = GT.astype('int8')
                val_f1.update(f1_score(GT, pred_lb).item(), images.size(0))
                val_acc.update(accuracy_score(GT, pred_lb).item(), images.size(0))
                val_sens.update(recall_score(GT, pred_lb).item(), images.size(0))
                val_prec.update(precision_score(GT, pred_lb).item(), images.size(0))
            if self.add_outputs:
                loss_ad1.update(loss1.item(), sp_wei.sum().item())
                loss_ad2.update(loss2.item(), sp_wei.sum().item())
                loss_ad3.update(loss3.item(), sp_wei.sum().item())

            batch_time.update(time.time() - end)
            end = time.time()
            lb_list.extend(GT)
            pd_list.extend(pred)


        if self.config.task == 'classification':
            labels = np.array(lb_list)
            preds = np.array(pd_list)
            preds = preds>0.5
            preds = preds*1
            print('testing set size', len(labels))
            print(f1_score(labels, preds))
            print(accuracy_score(labels, preds))
            print(recall_score(labels, preds))
            print(precision_score(labels, preds))
            return labels, preds, f1_score(labels, preds), accuracy_score(labels, preds), recall_score(labels, preds), precision_score(labels, preds)

class EarlyStopper:
    def __init__(self, patience=50, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

import scipy.stats as stats
import scipy
# Modeling with Numpy
def equation(a, b):
    """Return a 1D polynomial."""
    return np.polyval(a, b)

def plot_ci_manual(t, s_err, n, x, x2, y2, ax=None):
    """Return an axes of confidence bands using a simple approach.

    Notes
    -----
    .. math:: \left| \: \hat{\mu}_{y|x0} - \mu_{y|x0} \: \right| \; \leq \; T_{n-2}^{.975} \; \hat{\sigma} \; \sqrt{\frac{1}{n}+\frac{(x_0-\bar{x})^2}{\sum_{i=1}^n{(x_i-\bar{x})^2}}}
    .. math:: \hat{\sigma} = \sqrt{\sum_{i=1}^n{\frac{(y_i-\hat{y})^2}{n-2}}}

    References
    ----------
    .. [1] M. Duarte.  "Curve fitting," Jupyter Notebook.
       http://nbviewer.ipython.org/github/demotu/BMC/blob/master/notebooks/CurveFitting.ipynb

    """
    if ax is None:
        ax = plt.gca()

    ci = t * s_err * np.sqrt(1/n + (x2 - np.mean(x))**2 / np.sum((x - np.mean(x))**2))
    ax.fill_between(x2, y2 + ci, y2 - ci, color="#b9cfe7", edgecolor=None, alpha = 0.4)

    return ax


def plot_ci_bootstrap(xs, ys, resid, nboot=500, ax=None):
    """Return an axes of confidence bands using a bootstrap approach.

    Notes
    -----
    The bootstrap approach iteratively resampling residuals.
    It plots `nboot` number of straight lines and outlines the shape of a band.
    The density of overlapping lines indicates improved confidence.

    Returns
    -------
    ax : axes
        - Cluster of lines
        - Upper and Lower bounds (high and low) (optional)  Note: sensitive to outliers

    References
    ----------
    .. [1] J. Stults. "Visualizing Confidence Intervals", Various Consequences.
       http://www.variousconsequences.com/2010/02/visualizing-confidence-intervals.html

    """
    if ax is None:
        ax = plt.gca()

    bootindex = scipy.random.randint

    for _ in range(nboot):
        resamp_resid = resid[bootindex(0, len(resid) - 1, len(resid))]
        # Make coeffs of for polys
        pc = scipy.polyfit(xs, ys + resamp_resid, 1)
        # Plot bootstrap cluster
        ax.plot(xs, scipy.polyval(pc, xs), "b-", linewidth=2, alpha=3.0 / float(nboot))

    return ax

def plot_scatter(x, y, title, xlabel, ylabel, limi, spinexy = True, font = 35, markersize = 2, alpha = 1, perc = 0.95, hist = 0, xtic = 0, showr2 = 0):
    # plt.rcParams['font.family'] = 'Lucida Grande'
    x = np.array(x)
    y = np.array(y)

    def func(x, a, b):
        return a * x + b

    popt, pcov = curve_fit(func, x, y)
    # r2 = r2_score(np.array(x), np.array(y))
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    skr2 = r2_score(y, x)


    p, cov = np.polyfit(x, y, 1, cov=True)                     # parameters and covariance from of the fit of 1-D polynom.
    y_model = equation(p, x)
    n = y.size                                           # number of observations
    m = p.size                                                 # number of parameters
    dof = n - m                                                # degrees of freedom
    t = stats.t.ppf(perc, n - m)
    print(p)
    fig, ax = plt.subplots(figsize=(8, 8))
    # ax = fig.add_subplot(111)
    if spinexy:
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
    else:
        for key, spine in ax.spines.items():
            spine.set_visible(False)
    # Data
    # ax.plot(
    #     x, y, "o", color="#b9cfe7", markersize=8,
    #     markeredgewidth=1, markeredgecolor="b", markerfacecolor="None"
    # )
    from matplotlib import colors
    if hist:
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        counts, xedges, yedges, im = ax.hist2d(x, y, bins = 50, cmap = 'Blues', density =  1, norm=colors.LogNorm(), vmin = 0.00001)
        ax2 = plt.gca()
        divider = make_axes_locatable(ax2)
        cax = divider.append_axes("right", size="2.5%", pad=0.01)
        # cbar_ax = fig.add_axes([0.9, 0.96, 0.01, 0.8])
        # fig.colorbar(thplot, cax=cbar_ax, orientation="horizontal")
        cbar=fig.colorbar(im, cax = cax)
        # cbar=fig.colorbar(im, aspect = 50)
        tick_font_size = font
        cbar.outline.set_linewidth(0.15)
        cbar.ax.tick_params(labelsize=tick_font_size)
        # plt.axis('scaled')
    else:
        ax.scatter(x, y, color = 'teal', s=markersize)
    xx = [-30, limi]

    x2 = np.linspace(0, limi, 100)
    y2 = equation(p, x2)

    # Estimates of Error in Data/Model
    resid = y - y_model
    chi2 = np.sum((resid / y_model)**2)                        # chi-squared; estimates error in data
    chi2_red = chi2 / dof                                      # reduced chi-squared; measures goodness of fit
    s_err = np.sqrt(np.sum(resid**2) / dof)
    plot_ci_manual(t, s_err, n, x, x2, y2, ax=ax)

    # # Prediction Interval
    # pi = t * s_err * np.sqrt(1 + 1/n + (x2 - np.mean(x))**2 / np.sum((x - np.mean(x))**2))
    # ax.fill_between(x2, y2 + pi, y2 - pi, color="None", linestyle="--")
    # ax.plot(x2, y2 - pi, "--", color="0.5", label="95% Prediction Limits")
    # ax.plot(x2, y2 + pi, "--", color="0.5")
    plt.xlim(-30, limi)
    plt.ylim(-30, limi)

    plt.locator_params(axis='y', nbins=2)
    plt.locator_params(axis='x', nbins=2)
    if xtic:
        plt.xticks(fontsize=font)
    else:
        plt.xticks([])
    plt.yticks(fontsize=font)
    # plt.title(title, fontsize = 16)
    if showr2:
        ax.plot(np.array(xx), func(np.array(xx), *popt), 'teal', label='f(x) = %5.2f x + %5.2f\n$Pearson  r$ = %5.2f' % (popt[0], popt[1], r_value))
    else:
        ax.plot(np.array(xx), func(np.array(xx), *popt), 'teal', label='f(x) = %5.2f x + %5.2f' % (popt[0], popt[1]))

    ax.plot(xx, xx, '--', color = 'gray', alpha = alpha)
    # L = ax.legend(fontsize = 22,)
    # plt.setp(L.texts, family='DejaVu Sans')
    ax.legend(loc = 'upper left', fontsize = font-1, handlelength=0.5)
    if xtic:
        ax.yaxis.get_major_ticks()[0].label1.set_visible(False)

    plt.xlabel(xlabel,fontsize=font)
    plt.ylabel(ylabel,fontsize=font)

    return




def mae_group(intervals, preds, gtts):
    maes = []
    for i in range(len(intervals)-1):
        curgt = gtts[i]
        curpred = preds[i]
        assert len(curgt) == len(curpred)
        print(len(curgt))
        try:
            maei = mean_absolute_error(curgt, curpred)
            maes.append(maei)
        except:  # no higher than this range
            maes.append(0)

    return maes

def plot_box(ppd, gtt, gtts, preds, font = 35, title = None, spinexy = 0, lim =350):

    label_nas = ['0-50', '50-100', '100-150', '150-200', '200-250',' 250-300', '>300']
    # label_nas = ['1', '2', '3', '4', '5', '6', '7']
    labels = [0, 50, 100, 150, 200, 250, 300]
    # labels = [0, 1, 2, 3, 4, 5, 6]

    fig, ax = plt.subplots(figsize = (7, 7))

    ax.hist(gtt, bins=200, color='tan', alpha=0.4, density = 1, label = 'Reference')
    ax.locator_params(axis='y', nbins=4)
    quartiles = [2.5, 97.5]
    for q in np.percentile(gtt, quartiles):
        plt.axvline(q, ls = '--', lw = 3, color='firebrick', alpha = 0.6)
    ax.tick_params(axis='y')
    ax.hist(ppd, bins=200, color='#89979a', alpha=0.3, density = 1, label = 'Prediction')

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    # ax.set_ylabel('MSD')
    plt.xlim(-20, lim)
    # plt.xticks(rotation=45)

    # plt.locator_params(axis='x', nbins=3)
    plt.xticks(fontsize=font)
    plt.yticks(fontsize=font)
    # ax.set_title('Mean Squared Deviation (MSE) by height ranges')
    # ax.legend(bbox_to_anchor=(0.9,0.99), prop={'size': 12})
    # ax2.legend(bbox_to_anchor=(0.2,0.99), prop={'size': 12})

    ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis


    maes = []
    for i in range(7):

        maes.append(abs(gtts[i] - preds[i]))

    print(len(maes))
    # boxprops=dict( color='teal')
    boxprops=dict( color='chocolate', linewidth=5)
    line_props = dict(linestyle='--', linewidth = 3, color="grey", alpha=1)
    capprops = dict(linewidth = 5, color="grey", alpha = 1)
    medianprops = dict(linewidth=5, color = 'purple')
    # meanlineprops = dict(linestyle='solid', linewidth=3.5, color='darkorange')
    meanlineprops = dict(linestyle='solid', linewidth=5, color='green')
    bp = ax2.boxplot( maes, widths = 20, showfliers=False, showmeans = 1,  meanprops=meanlineprops, meanline=True, patch_artist=True, whiskerprops=line_props, boxprops = boxprops, capprops = capprops, medianprops=medianprops, positions = labels)
    # ax_box_x.set_title('Max height per tree - Errors at 10m height intervals', fontsize = 16)
    # fill with colors

    for patch in bp['boxes']:
        # patch.set_facecolor(color)
        # patch.set(facecolor = 'cadetblue', alpha = 0.6 )
        patch.set(facecolor = 'chocolate', alpha = 0.6 )
    ax2.set_xticks(labels)
    ax2.set_xticklabels(label_nas)

    ax.yaxis.set_ticks_position('right')
    ax2.yaxis.set_ticks_position('left')
    # ax.set_yticklabels(fontsize=15)
    ax.yaxis.set_tick_params(labelsize=font)
    ax2.yaxis.set_tick_params(labelsize=font)
    ax2.locator_params(axis='y', nbins=4)
    if spinexy:
        # ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        # ax2.spines['right'].set_visible(False)
        ax2.spines['top'].set_visible(False)
    else:
        for key, spine in ax.spines.items():
            spine.set_visible(False)
        for key, spine in ax2.spines.items():
            spine.set_visible(False)
    # ax2.legend(loc ='upper right', prop={'size': 12})
    # ax.legend(loc ='upper left', prop={'size': 26})
    # ax3 = ax2.twinx()
    # ax3.bar(labels, sbs, width = 1,  label='SB', alpha = 1, color = 'steelblue')

    plt.show()

    return


def hex_to_rgb_color_list(colors):
    """
    Take color or list of hex code colors and convert them
    to RGB colors in the range [0,1].
    Parameters:
        - colors: Color or list of color strings of the format
                  '#FFF' or '#FFFFFF'
    Returns:
        The color or list of colors in RGB representation.
    """
    if isinstance(colors, str):
        colors = [colors]

    for i, color in enumerate(
        [color.replace('#', '') for color in colors]
    ):
        hex_length = len(color)

        if hex_length not in [3, 6]:
            raise ValueError(
                'Colors must be of the form #FFFFFF or #FFF'
            )

        regex = '.' * (hex_length // 3)
        colors[i] = [
            int(val * (6 // hex_length), 16) / 255
            for val in re.findall(regex, color)
        ]

    return colors[0] if len(colors) == 1 else colors


def blended_cmap(rgb_color_list):
    """
    Created a colormap blending from one color to the other.
    Parameters:
        - rgb_color_list: A list of colors represented as [R, G, B]
          values in the range [0, 1], like [[0, 0, 0], [1, 1, 1]],
          for black and white, respectively.
    Returns:
        A matplotlib `ListedColormap` object
    """
    if not isinstance(rgb_color_list, list):
        raise ValueError('Colors must be passed as a list.')
    elif len(rgb_color_list) < 2:
        raise ValueError('Must specify at least 2 colors.')
    elif (
        not isinstance(rgb_color_list[0], list)
        or not isinstance(rgb_color_list[1], list)
    ) or (
        len(rgb_color_list[0]) != 3 or len(rgb_color_list[1]) != 3
    ):
        raise ValueError(
            'Each color should be represented as a list of size 3.'
        )

    N, entries = 256, 4 # red, green, blue, alpha
    rgbas = np.ones((N, entries))

    segment_count = len(rgb_color_list) - 1
    segment_size = N // segment_count
    remainder = N % segment_count # need to add this back later

    for i in range(entries - 1): # we don't alter alphas
        updates = []
        for seg in range(1, segment_count + 1):
            # determine how much needs to be added back to account for remainders
            offset = 0 if not remainder or seg > 1 else remainder

            updates.append(np.linspace(
                start=rgb_color_list[seg - 1][i],
                stop=rgb_color_list[seg][i],
                num=segment_size + offset
            ))

        rgbas[:,i] = np.concatenate(updates)

    return ListedColormap(rgbas)


def draw_cmap(cmap, values=np.array([[0, 1]]), **kwargs):
    """
    Draw a colorbar for visualizing a colormap.
    Parameters:
        - cmap: A matplotlib colormap
        - values: The values to use for the colormap, defaults to [0, 1]
        - kwargs: Keyword arguments to pass to `plt.colorbar()`
    Returns:
        A matplotlib `Colorbar` object, which you can save with:
        `plt.savefig(<file_name>, bbox_inches='tight')`
    """
    img = plt.imshow(values, cmap=cmap)
    cbar = plt.colorbar(**kwargs)
    img.axes.remove()
    return cbar



def load_data(ppd, gtt, sample = 0):
    ppd = np.nan_to_num(ppd)
    gtt = np.nan_to_num(gtt)
    gtm = int(np.ceil(gtt.max()))

    inds = []
    intervals = [0, 50, 100, 150, 200, 250, 300, gtm]
    for i in range(len(intervals)-1):
        indi = [idx for idx,val in enumerate(gtt) if intervals[i] <= val < intervals[i+1]]
        inds.append(indi)


    preds = []
    gtts = []
    for i in range(len(intervals)-1):
        predi = ppd[inds[i]]
        preds.append(predi)
        gtti = gtt[inds[i]]
        gtts.append(gtti)

    return ppd, gtt, preds, gtts, intervals, gtm


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        # self.avg = self.sum / self.count
        self.avg = div0(self.sum,self.count)

def get_metrics(metrics, stage):
    """ Returns a dictionary of all metrics and losses being tracked
    """
    met = {}
    met[f"{stage}_rmse"] = metrics['rmse']
    met[f"{stage}_loss"] = metrics['loss']
    met[f"{stage}_r2"] = metrics['r2']

    # metrics[f"{self._stage}_total_rmse"] = self._rmse[-1].value()
    # metrics[f"{self._stage}_total_mae"] = self._mae[-1].value()
    return met

def get_metrics_classification(metrics, stage):
    """ Returns a dictionary of all metrics and losses being tracked
    """
    met = {}
    met[f"{stage}_f1"] = metrics['f1']
    # met[f"{stage}_acc"] = metrics['acc']
    met[f"{stage}_sens"] = metrics['sens']

    # metrics[f"{self._stage}_total_rmse"] = self._rmse[-1].value()
    # metrics[f"{self._stage}_total_mae"] = self._mae[-1].value()
    return met


def div0(x,y):
    try:
        return x/y
    except ZeroDivisionError:
        return 0




def focal_loss(logits, labels, alpha=None, gamma=2):
    """Compute the focal loss between `logits` and the ground truth `labels`.
    Focal loss = -alpha_t * (1-pt)^gamma * log(pt)
    where pt is the probability of being classified to the true class.
    pt = p (if true class), otherwise pt = 1 - p. p = sigmoid(logit).
    Args:
      logits: A float tensor of size [batch, num_classes].
      labels: A float tensor of size [batch, num_classes].
      alpha: A float tensor of size [batch_size]
        specifying per-example weight for balanced cross entropy.
      gamma: A float scalar modulating loss from hard and easy examples.
    Returns:
      focal_loss: A float32 scalar representing normalized total loss.
    """
    bc_loss = F.binary_cross_entropy_with_logits(input=logits, target=labels, reduction="none")

    if gamma == 0.0:
        modulator = 1.0
    else:
        modulator = torch.exp(-gamma * labels * logits - gamma * torch.log(1 + torch.exp(-1.0 * logits)))

    loss = modulator * bc_loss

    if alpha is not None:
        weighted_loss = alpha * loss
        focal_loss = torch.sum(weighted_loss)
    else:
        focal_loss = torch.sum(loss)

    focal_loss /= torch.sum(labels)
    return focal_loss


class Loss(torch.nn.Module):
    def __init__(
        self,
        loss_type: str = "cross_entropy",
        beta: float = 0.999,
        fl_gamma=2,
        samples_per_class=None,
        class_balanced=False,
    ):
        """
        Compute the Class Balanced Loss between `logits` and the ground truth `labels`.
        Class Balanced Loss: ((1-beta)/(1-beta^n))*Loss(labels, logits)
        where Loss is one of the standard losses used for Neural Networks.
        reference: https://openaccess.thecvf.com/content_CVPR_2019/papers/Cui_Class-Balanced_Loss_Based_on_Effective_Number_of_Samples_CVPR_2019_paper.pdf
        Args:
            loss_type: string. One of "focal_loss", "cross_entropy",
                "binary_cross_entropy", "softmax_binary_cross_entropy".
            beta: float. Hyperparameter for Class balanced loss.
            fl_gamma: float. Hyperparameter for Focal loss.
            samples_per_class: A python list of size [num_classes].
                Required if class_balance is True.
            class_balanced: bool. Whether to use class balanced loss.
        Returns:
            Loss instance
        """
        super(Loss, self).__init__()

        if class_balanced is True and samples_per_class is None:
            raise ValueError("samples_per_class cannot be None when class_balanced is True")

        self.loss_type = loss_type
        self.beta = beta
        self.fl_gamma = fl_gamma
        self.samples_per_class = samples_per_class
        self.class_balanced = class_balanced

    def forward(
        self,
        logits: torch.tensor,
        labels: torch.tensor,
    ):
        """
        Compute the Class Balanced Loss between `logits` and the ground truth `labels`.
        Class Balanced Loss: ((1-beta)/(1-beta^n))*Loss(labels, logits)
        where Loss is one of the standard losses used for Neural Networks.
        Args:
            logits: A float tensor of size [batch, num_classes].
            labels: An int tensor of size [batch].
        Returns:
            cb_loss: A float tensor representing class balanced loss
        """

        batch_size = logits.size(0)
        num_classes = 1
        labels_one_hot = labels

        if self.class_balanced:
            effective_num = 1.0 - np.power(self.beta, self.samples_per_class)
            weights = (1.0 - self.beta) / np.array(effective_num)
            weights = weights / np.sum(weights) * num_classes
            weights = torch.tensor(weights, device=logits.device).float()

            if self.loss_type != "cross_entropy":
                weights = weights.unsqueeze(0)
                weights = weights.repeat(batch_size, 1) * labels_one_hot
                weights = weights.sum(1)
                weights = weights.unsqueeze(1)
                weights = weights.repeat(1, num_classes)
        else:
            weights = None

        if self.loss_type == "focal_loss":
            cb_loss = focal_loss(logits, labels_one_hot, alpha=weights, gamma=self.fl_gamma)
        elif self.loss_type == "cross_entropy":
            cb_loss = F.cross_entropy(input=logits, target=labels_one_hot, weight=weights)
        elif self.loss_type == "binary_cross_entropy":
            cb_loss = F.binary_cross_entropy_with_logits(input=logits, target=labels_one_hot, weight=weights)
        elif self.loss_type == "softmax_binary_cross_entropy":
            pred = logits.softmax(dim=1)
            cb_loss = F.binary_cross_entropy(input=pred, target=labels_one_hot, weight=weights)
        return cb_loss
