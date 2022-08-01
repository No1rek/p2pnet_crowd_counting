import torch
import torch.nn as nn
import numpy as np
from time import time
import copy
import os
from .datasets import DataLoader
import random
import math
from torch.utils.tensorboard import SummaryWriter


class ModelTrainer:
    def __init__(self, config:dict={}):
        """
            Performs model training using AdamW, StepLRScheduler, Tensorboard Summary writer and early stopping

            Params:
            config - overrides object attributes, SummaryWriter adds only overriden parameters to run name

            Methods:
            train - run model training
        """

        # general
        self.tensorboard_dir = "./runs"
        self.checkpoints_dir = "./checkpoints"
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.seed = 42

        # model - read model docstring
        self.hidden_size = 256
        self.n_anchors = 4

        # criterion - read criterion docstring
        self.eos_coef = 0.5
        self.cost_point = 0.05
        self.reg_loss_weight = 2e-4

        # optimization
        self.lr = 1e-4
        self.lr_backbone = 1e-5
        self.lr_drop_epoch = 3500
        self.lr_drop_gamma = 0.1
        self.max_grad_norm = 0.1 # gradient clipping
        self.weight_decay = 1e-4

        # training
        self.batch_size = 8
        self.n_epochs = 3500
        self.eval_freq = 5
        self.patience = 50 # early stopping
        
        # keep only manual parameters
        self.manual_config = {k:v for k,v in config.items() if getattr(self, k) != v}
        for k,v in config.items():
            setattr(self, k, v)
            
    def train(self, model_cls, criterion_cls, train_set, val_set, run_params={}, comment=""):
        self.seed_everything()
        start = time()

        model = model_cls(self.n_anchors, self.hidden_size, device=self.device)
        best_model = model_cls(self.n_anchors, self.hidden_size, device=self.device)

        run_name = self.make_writer_suffix(self.manual_config, run_params, comment)
        print(run_name)
        
        criterion = criterion_cls(eos_coef=self.eos_coef, cost_point=self.cost_point, 
                                reg_loss_weight=self.reg_loss_weight, device=self.device)

        # define optimizers: different lr for backbone
        param_dicts = [
                {
                    "params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad],
                    "lr":self.lr,
                    "weight_decay":self.weight_decay,
                },
                {
                    "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
                    "lr": self.lr_backbone,
                    "weight_decay":self.weight_decay
                },
        ]
        optimizer = torch.optim.AdamW(param_dicts, lr=self.lr, weight_decay=self.weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 
                                                        step_size=self.lr_drop_epoch,
                                                        gamma=self.lr_drop_gamma)

        # create samplers
        sampler_train = torch.utils.data.RandomSampler(train_set)
        sampler_val = torch.utils.data.SequentialSampler(val_set)
        batch_sampler_train = torch.utils.data.BatchSampler(sampler_train,
                                                            self.batch_size,
                                                            drop_last=False)
        batch_sampler_val = torch.utils.data.BatchSampler(sampler_val,
                                                          self.batch_size, 
                                                          drop_last=True)
        dataloader_train = DataLoader(train_set, 
                                batch_sampler=batch_sampler_train, 
                                collate_fn=self.collate_fn)
        dataloader_val = DataLoader(val_set, 
                                batch_sampler=batch_sampler_val,
                                collate_fn=self.collate_fn)  

        with SummaryWriter(os.path.join(self.tensorboard_dir, run_name)) as writer:
            best_loss = np.inf
            best_epoch = 1
            for epoch in range(1, self.n_epochs):
                loss_train = self.train_step(model, criterion, dataloader_train, 
                                             optimizer, lr_scheduler)
                print(f"[{epoch}] train: {loss_train:.4} best_eval: {best_loss:.4}")
                if epoch % self.eval_freq == 0:
                    loss_eval = self.eval_step(model, criterion, dataloader_val)

                    writer.add_scalars("loss", {
                        "train":loss_train,
                        "eval":loss_eval
                    }, epoch)

                    # early stopping
                    if best_loss > loss_eval:
                        best_loss = loss_eval
                        best_epoch = epoch
                        best_model.load_state_dict(copy.deepcopy(model.state_dict()))
                    elif epoch - best_epoch > self.patience:
                        break
        print(f"time:{time() - start:.1f} best loss:{best_loss:.4f} best epoch:{best_epoch}")
        return best_model
        
    def train_step(self, model, criterion, dataloader, optimizer, lr_scheduler):
        model.train()
        model.eval()
        running_loss = 0.0
        n_iters = 0
        for i,(features, labels) in enumerate(dataloader):
            # forward pass
            optimizer.zero_grad()
            preds = model(features)
            loss = criterion(preds, labels)
            loss_value = loss.item()
            if not math.isfinite(loss_value): 
                raise Exception(f"Loss is {loss_value}")

            # backward pass
            loss.backward()
            if (self.max_grad_norm > 0):
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.max_grad_norm)
            optimizer.step()

            running_loss += loss_value
            n_iters += 1   
        lr_scheduler.step()
        avg_loss = running_loss / n_iters
        return avg_loss

    @torch.no_grad()
    def eval_step(self, model, criterion, dataloader):
        model.eval()
        criterion.eval()
        running_loss = 0.0
        n_iters = 0
        for i,(features, labels) in enumerate(dataloader):
            preds = model(features)
            loss = criterion(preds, labels)
            loss_value = loss.item()
            running_loss += loss_value
            n_iters += 1
        avg_loss = running_loss / n_iters
        return avg_loss
        
    def seed_everything(self):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)
        
    def collate_fn(self, data):
        """
            Return types:
            features - torch.Tensor
            labels - list of torch.Tensor's
        """
        transposed_data = list(zip(*data))
        features  = torch.stack(transposed_data[0], 0).to(self.device)
        labels  = [l.to(self.device) for l in transposed_data[1]]
        return features, labels
    
    def make_writer_suffix(self, trainer_params, run_params, comment):
        """
            Generates run name
        """
        timestr = str(int(time()))
        params = {**trainer_params, **run_params}
        if len(comment) > 0:
            comment = '_' + comment
        if len(params) > 0:
            param_str = '_'+str(params)[1:-1].replace(' ', '').replace(',', '_').replace("'", '').replace(":", " ")
        else: param_str = ""
        return timestr+comment+param_str