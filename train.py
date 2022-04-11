#Author LCD
from utils.utils import parameters_read, resize
from utils.torch_utils import select_device
import os
from tensorboardX import SummaryWriter
import json
from utils import transform as tr
from utils.load_data import load_data_from_path as load_data
from torchvision import transforms
from networks.u2net import U2NETP
from torch import nn
import torch
import traceback
from utils.loss import loss_function, IOU
from utils.sam import SharpnessAwareMinimization
import numpy as np
import nni
from tqdm.autonotebook import tqdm
from torch.utils.data import DataLoader


class train_model(object):
    def __init__(self, opt):
        print("Model Name:", opt.model)
        print("Dataset:", opt.dataset)

        self.model_name = opt.model
        self.data_name = opt.dataset
        self.learning_rate = opt.lr
        self.opti = opt.optimizer.upper()


        #Load model Params
        self.model_params = parameters_read(f'configs/model_params/{self.model_name}.yml')
        self.device = select_device(opt.device, self.model_params.params['hyperparams']['batch_size'])
        self.img_size = self.model_params.params['img_size']
        self.batch_size = self.model_params.params['hyperparams']['batch_size']
        self.num_worker = self.model_params.params['hyperparams']['num_workers']

        # Create Logs dir and checkpoints
        exp_name, trial_name = os.environ['NNI_OUTPUT_DIR'].split('\\')[-2:]
        # exp_name = 'trials'
        # trial_name = 'testcode'

        trial_run = '{}_{}_{}'.format(self.model_name,exp_name, trial_name)
        save_path = 'runs/{}/'.format(trial_run)
        self.save_log = save_path + 'logs/'
        self.save_params = save_path + 'params/'
        self.save_cp = save_path + 'cps/'
        os.makedirs(self.save_log, exist_ok=True)
        os.makedirs(self.save_params, exist_ok=True)
        os.makedirs(self.save_cp, exist_ok=True)
        with open('{}/args.txt'.format(self.save_params), 'w') as f:
            json.dump(opt.__dict__, f, indent=2)
        self.writer = SummaryWriter(self.save_log)
        print('Saving Location: {}'.format(save_path))

        #Load dataset
        self.dataset = parameters_read(f'data/{self.data_name}.yaml')
        self.num_class = self.dataset.params['nc']
        self.root_dir = self.dataset.params['root_dir']

        train_transforms = [
            tr.augmentation_duong(
                rotate=0.2,
                scale=0.2,
                scale_range=[1.1, 1.5, 0.1],
                flipx=0.2,
                flipy=0.2,
                enhance_contrast=0.2,
                sliding=0.2
            ),
            tr.Normalizer(),
            tr.Resizer(self.img_size)
        ]
        train_set = load_data(root_dir=self.root_dir, num_class=self.num_class, dataset='train', transform=transforms.Compose(train_transforms))
        train_params = {
            'batch_size': self.batch_size,
            'shuffle': True,
            'drop_last': True,
            'num_workers': self.num_worker
        }

        self.training_generator = DataLoader(train_set, **train_params)


        val_transforms = [
            tr.Normalizer(),
            tr.Resizer(self.img_size)
        ]
        val_set = load_data(root_dir=self.root_dir, num_class=self.num_class, dataset='val',
                                 transform=transforms.Compose(val_transforms))

        val_params = {
            'batch_size': 1,
            'shuffle': False,
            'drop_last': False,
            'num_workers': self.num_worker
        }
        self.val_generator = DataLoader(val_set, **val_params)

        #Load model
        self.channels = self.model_params.u2netparams['channels']
        model = U2NETP(self.channels, self.num_class)
        self.model = ModelWithLoss(model=model, loss_func_type=0)
        self.model = self.model.to(self.device)

        #Train Init
        if self.opti == 'SGD':
            base_optimizer = torch.optim.SGD  # ()
            self.optimizer = SharpnessAwareMinimization(self.model.parameters(), base_optimizer, lr=self.learning_rate, momentum=0.9)
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=self.optimizer, T_0=10, T_mult=2)
        elif self.opti == "ADAMW":
            self.optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=3, verbose=True)

        self.best_loss = 1e5
        self.best_acc = 0
        self.best_epoch = 0
        self.step = 0
        self.num_iter_per_epoch = len(self.training_generator)
        self.save_interval = self.num_iter_per_epoch
        self.num_epochs = opt.epochs
        self.res = []
        self.valid_losses_per_epoch = []
        self.valid_accuracy_per_epoch = []


    def train(self, epoch):
        self.model.train()
        last_epoch = self.step // self.num_iter_per_epoch
        epoch_loss = []
        epoch_loss2 = []
        progress_bar = tqdm(self.training_generator)

        for iter, data in enumerate(progress_bar):
            if iter < self.step - last_epoch * self.num_iter_per_epoch:
                progress_bar.update()
                continue
            try:

                imgs, mask = data['img'], data['mask']
                imgs = imgs.cuda()
                imgs = imgs.permute(0, 3, 1, 2)
                # imgs = torch.unsqueeze(imgs, 0)

                mask = mask.cuda()
                mask = mask.permute(0, 3, 1, 2)
                # mask = torch.unsqueeze(mask, 0)

                if self.opti == "SGD":
                    self.scheduler.step(epoch + iter / self.num_iter_per_epoch)
                self.optimizer.zero_grad()
                loss, _ = self.model(imgs, mask, istraining=True)

                descriptor = '[Train] Step: {}. Epoch: {}/{}. Iteration: {}/{}. '.format(
                        self.step, epoch, self.num_epochs, iter + 1, self.num_iter_per_epoch)

                epoch_loss.append(loss.item())
                self.writer.add_scalars('Loss', {'train': loss.item()}, self.step)
                if loss == 0 or not torch.isfinite(loss):
                    continue
                loss.backward()
                if self.opti == "SGD":
                    self.optimizer.first_step(zero_grad=True)
                else:
                    self.optimizer.step()
                loss2, _ = self.model(imgs, mask, istraining=True)

                epoch_loss2.append(loss2.item())
                self.writer.add_scalars('Loss', {'train_2nd': loss2.item()}, self.step)
                loss2.backward()
                if self.opti == "SGD":
                    self.optimizer.second_step(zero_grad=True)
                else:
                    self.optimizer.step()
                progress_bar.set_description(descriptor)

                current_lr = self.optimizer.param_groups[0]['lr']
                self.writer.add_scalar('learning_rate', current_lr, self.step)
                self.step += 1

            except Exception as e:
                print('[Error]', traceback.format_exc())
                print(e)
                continue

    def validation(self, epoch):
        self.model.eval()
        epoch_loss = []
        epoch_acc = []
        progress_bar = tqdm(self.val_generator)

        for iter, data in enumerate(progress_bar):
            with torch.no_grad():
                if isinstance(data, dict):
                    imgs = data['img']
                    mask = data['mask']
                    # filename = data['filename']
                else:
                    imgs, mask = data

                try:
                    imgs = imgs.cuda()
                    imgs = imgs.permute(0, 3, 1, 2)
                    # imgs = torch.unsqueeze(imgs, 0)
                    # annot = annot.cuda()
                    mask = mask.cuda()
                    # cls = self.model(imgs)
                    mask = mask.permute(0, 3, 1, 2)
                    # mask = torch.unsqueeze(mask, 0)

                    loss, iou = self.model(imgs, mask)
                    descriptor = '[Valid] Step: {}. Epoch: {}/{}. Iteration: {}/{}. '.format(
                        epoch * len(progress_bar) + iter, epoch, self.num_epochs, iter + 1, len(progress_bar))
                    epoch_loss.append(loss.item())
                    epoch_acc.append(iou.item())
                    progress_bar.set_description(descriptor)
                except Exception as e:
                    print(e)


        mean_loss = np.mean(epoch_loss)
        mean_acc = np.mean(epoch_acc)

        self.valid_losses_per_epoch.append(mean_loss)
        self.valid_accuracy_per_epoch.append(mean_acc)

        self.writer.add_scalars('Loss', {'val': mean_loss.item()}, self.step)
        self.writer.add_scalars('Epoch_Loss', {'val': mean_loss.item()}, epoch)
        self.writer.add_scalars('Acc', {'val': mean_acc.item()}, epoch)

        val_descrip = '[Valid] Epoch: {}. Loss: {}. Acc: {}'.format(epoch, mean_loss, mean_acc)
        print(val_descrip)

        nni.report_intermediate_result(mean_acc)
        if epoch == self.num_epochs -1:
            max_epoch = np.argmax(self.valid_accuracy_per_epoch)
            max_acc = self.valid_accuracy_per_epoch[max_epoch]
            nni.report_final_result({int(max_epoch): max_acc})

        if self.best_acc < mean_acc:
            self.best_acc = mean_acc
            # save_checkpoint(self.model, self)
            save_checkpoint(self.model, self.save_cp,
                            '/{}_{}.pth'.format(epoch, self.step))

    def start(self):
        for epoch in range(self.num_epochs):
            self.train(epoch)
            self.validation(epoch)

    def closure(self, loss):
        return loss



class ModelWithLoss(nn.Module):
    def __init__(self, model, debug=False, loss_func_type=0):
        super().__init__()
        self.model = model
        self.criterion = loss_function(loss_func_type, 'mean') #losses[loss_func_type](**loss_args[loss_func_type])
        self.debug = debug

    def forward(self, imgs, mask, istraining=False, **kwargs):
        output = self.model(imgs)

        B, C, H, W = output.shape
        # oH, oW = annotations.shape[1:]
        oH, oW = mask.shape[2:]
        # oH, oW = annotations.shape[2:]

        if H != oH or W != oW:
            seg_logit = resize(input=output, size=mask.shape[2:], mode='bilinear', align_corners=True)
            # seg_logit = resize(input=output, size=annotations.shape[2:], mode='bilinear', align_corners=True)
        else:
            seg_logit = output

        acc = None
        losses = self.criterion(seg_logit, mask)

        acc = IOU(output, mask)

        return losses, acc

def save_checkpoint(model, saved_path, name):
    torch.save(model.model.state_dict(), saved_path + name)