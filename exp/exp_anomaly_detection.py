import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy('file_system')
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
import math
import json
from models.CrossAD.Basic_CrossAD import Basic_CrossAD
from models.Left.LEFT import Left

warnings.filterwarnings('ignore')


class Exp_Anomaly_Detection():
    def __init__(self, args, id=None):
        self.args = args
        self.device = self._acquire_device()

        if self.args.data_origin == "UCR":
            self.model_configs = Configs(
                os.path.join(self.args.configs_path, self.args.data_origin, f"model_configs_{id}.json"))
            self.model_configs.device = self.device
            self.train_configs = Configs(
                os.path.join(self.args.configs_path, self.args.data_origin, "train_configs.json"))

            if self.args.model_class in ['Left', ]:
                self.hyper_args = Configs(
                    os.path.join(self.args.configs_path, self.args.data, f"model_configs_Left.json"))

        else:
            self.model_configs = Configs(
                os.path.join(self.args.configs_path, self.args.data, f"model_configs_{id}.json"))
            self.model_configs.device = self.device
            self.train_configs = Configs(os.path.join(self.args.configs_path, self.args.data, "train_configs.json"))

            if self.args.model_class in ['Left', ]:
                self.hyper_args = Configs(
                    os.path.join(self.args.configs_path, self.args.data, f"model_configs_Left.json"))

        self.model = self._build_model().to(self.device)

        if self.args.data_origin == "UCR":
            self.model_save_path = os.path.join(self.args.model_save_path, self.args.data_origin, self.args.model_class,
                                                f"{self.args.data}/checkpoints_{id}")  # save model checkpoints
            self.rst_save_path = os.path.join(self.args.results_save_path, self.args.data_origin, self.args.model_class,
                                              f"{self.args.data}/results_{id}")  # save results
        else:
            self.model_save_path = os.path.join(self.args.model_save_path, self.args.data, self.args.model_class,
                                                f"checkpoints_{id}")  # save model checkpoints
            self.rst_save_path = os.path.join(self.args.results_save_path, self.args.data, self.args.model_class,
                                              f"results_{id}")  # save results

        if not os.path.exists(self.model_save_path):
            os.makedirs(self.model_save_path)
        if not os.path.exists(self.rst_save_path):
            os.makedirs(self.rst_save_path)

    def _build_model(self):
        model_class = eval(self.args.model_class)
        if self.args.model_class in ['Left', ]:
            model = model_class(self.hyper_args)
        else:
            model = model_class(self.model_configs)
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _basic(self):
        import pandas as pd
        from thop import profile
        from thop import clever_format
        dummy_input = torch.rand(1, self.model_configs.seq_len, self.model_configs.enc_in).to(self.device)
        flops, params = profile(self.model, inputs=(dummy_input, None, None, None))
        flops, params = clever_format([flops, params], '%.3f')
        basic_info = pd.DataFrame([{'params': params, 'flops': flops}])
        basic_info.to_csv(self.model_save_path + '/basic_info.csv', index=False)
        print(basic_info)

    def _acquire_device(self):
        if self.args.use_gpu and self.args.gpu_type == 'cuda':
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        elif self.args.use_gpu and self.args.gpu_type == 'mps':
            device = torch.device('mps')
            print('Use GPU: mps')
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _get_data(self, flag, step=None):
        if self.args.data_origin == "DADA":
            from data_provider.data_provider import data_provider
        elif self.args.data_origin == "UCR":
            from data_provider.data_provider_UCR import data_provider
        win_size = self.model_configs.seq_len
        if step is None:
            step = win_size
        batch_size = self.train_configs.batch_size
        data_set, data_loader = data_provider(
            root_path=self.args.root_path,
            datasets=self.args.data,
            batch_size=batch_size,
            win_size=win_size,
            step=step,
            flag=flag,
        )
        return data_set, data_loader

    def _select_optimizer(self):
        if self.train_configs.optim == "adam":
            model_optim = optim.Adam(self.model.parameters(), lr=self.train_configs.learning_rate)
        elif self.train_configs.optim == "adamw":
            model_optim = optim.AdamW(self.model.parameters(), lr=self.train_configs.learning_rate)
        return model_optim

    def _adjust_learning_rate(self, optimizer, epoch, train_configs, verbose=True, **other_args):
        if train_configs.lradj == 'type1':
            lr_adjust = {epoch: train_configs.learning_rate * (0.5 ** ((epoch - 1) // 1))}
        elif train_configs.lradj == 'type2':
            lr_adjust = {2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6, 10: 5e-7, 15: 1e-7, 20: 5e-8}
        elif train_configs.lradj == 'type3':
            lr_adjust = {epoch: train_configs.learning_rate if epoch < 3 else train_configs.learning_rate * (
                        0.9 ** ((epoch - 3) // 1))}
        elif train_configs.lradj == "cosine":
            lr_adjust = {epoch: train_configs.learning_rate / 2 * (
                        1 + math.cos(epoch / train_configs.train_epochs * math.pi))}
        elif train_configs.lradj == '1cycle':
            scheduler = other_args['scheduler']
            lr_adjust = {epoch: scheduler.get_last_lr()[0]}

        if epoch in lr_adjust.keys():
            lr = lr_adjust[epoch]
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            if verbose: print('Updating learning rate to {}'.format(lr))

    def vali(self, vali_loader):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, _) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                ms_loss, q_latent_distance = self.model(batch_x, None, None, None)
                loss = ms_loss
                total_loss.append(loss.item())
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self):
        self._basic()

        _, train_loader = self._get_data(flag='train', step=1)
        _, vali_loader = self._get_data(flag='val')
        train_steps = len(train_loader)

        model_optim = self._select_optimizer()
        early_stopping = EarlyStopping(patience=self.train_configs.patience, verbose=True)
        if self.train_configs.lradj == "1cycle":
            scheduler = optim.lr_scheduler.OneCycleLR(optimizer=model_optim,
                                                      steps_per_epoch=train_steps,
                                                      pct_start=self.train_configs.pct_start,
                                                      epochs=self.train_configs.train_epochs,
                                                      max_lr=self.train_configs.learning_rate
                                                      )

        time_now = time.time()
        for epoch in range(self.train_configs.train_epochs):
            iter_count = 0
            train_loss = []

            epoch_time = time.time()
            for i, (batch_x, batch_y) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                ms_loss, q_latent_distance = self.model(batch_x, None, None, None)
                loss = ms_loss
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.train_configs.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                model_optim.step()

                if self.train_configs.lradj == '1cycle':
                    self._adjust_learning_rate(model_optim, epoch + 1, self.train_configs, verbose=False,
                                               scheduler=scheduler)
                    scheduler.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_loader)
            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(epoch + 1, train_steps,
                                                                                           train_loss, vali_loss))

            early_stopping(vali_loss, self.model, self.model_save_path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            if self.train_configs.lradj != "1cycle":
                self._adjust_learning_rate(model_optim, epoch + 1, self.train_configs)
            else:
                print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))

        best_model_path = self.model_save_path + '/checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

    def test(self, **args):
        _, test_loader = self._get_data(flag='test')

        print('loading model...', end='')
        self.model.load_state_dict(torch.load(os.path.join(self.model_save_path, 'checkpoint.pth')), strict=False)
        print('done')

        self.model.eval()
        with torch.no_grad():
            # test set
            attens_energy = []
            test_labels = []
            # test_series = []
            for i, (batch_x, batch_y) in enumerate(test_loader):
                # test_series.append(batch_x)
                test_labels.append(batch_y)

                batch_x = batch_x.float().to(self.device)
                score, q_latent_distance = self.model.infer(batch_x, None, None, None)
                score = score.detach().cpu().numpy()
                attens_energy.append(score)

            attens_energy = np.concatenate(attens_energy, axis=0)  # nb x t x c
            attens_energy = attens_energy.reshape(-1, attens_energy.shape[-1])  # nb*t x c
            test_energy = np.array(attens_energy)
            test_labels = np.concatenate(test_labels, axis=0).reshape(-1)  # nb*t
            test_labels = np.array(test_labels)
            test_gt = test_labels.astype(int)
            # test_series = np.concatenate(test_series, axis=0)                         # nb x t x c
            # test_series = test_series.reshape(-1, test_series.shape[-1])              # nb*t x c
            # test_series = np.array(test_series)

        np.save(self.rst_save_path + '/a_gt.npy', test_gt)  # nb*t
        np.save(self.rst_save_path + '/a_test_energy.npy', test_energy)  # nb*t x c

        test_energy = np.mean(test_energy, axis=-1)  # nb*t

        from ts_ad_evaluation import Evaluator
        evaluator = Evaluator(test_gt, test_energy, self.rst_save_path)
        res = evaluator.evaluate(metrics=self.args.metrics)  # not need threshold
        return res

    def evaluate_spot(self, **args):
        gt = np.load(self.rst_save_path + '/a_gt.npy')
        test_energy = np.load(self.rst_save_path + '/a_test_energy.npy')

        test_energy = np.mean(test_energy, axis=-1)  # nb*t

        from ts_ad_evaluation import Evaluator
        evaluator = Evaluator(gt, test_energy, self.rst_save_path)

        # find threshold by spot
        _, init_loader = self._get_data(flag='init')

        print('loading model...', end='')
        self.model.load_state_dict(torch.load(os.path.join(self.model_save_path, 'checkpoint.pth')), strict=False)
        print('done')

        self.model.eval()
        with torch.no_grad():
            attens_energy = []
            for i, (batch_x, batch_y) in enumerate(init_loader):
                batch_x = batch_x.float().to(self.device)
                score, q_latent_distance = self.model.infer(batch_x, None, None, None)
                score = score.detach().cpu().numpy()
                attens_energy.append(score)

            attens_energy = np.concatenate(attens_energy, axis=0)  # nb x t x c
            attens_energy = attens_energy.reshape(-1, attens_energy.shape[-1])  # nb*t x c
            init_energy = np.array(attens_energy)

        init_energy = np.mean(init_energy, axis=-1)  # nb*t

        thresholds = evaluator.find_thres(method="spot", init_score=init_energy, q=args['t'])

        evaluator.evaluate(metrics=['affiliation'], affiliation=thresholds)

    def evaluate_UCR_accomplish(self, **args):
        _, test_loader = self._get_data(flag='test')

        print('loading model...', end='')
        self.model.load_state_dict(torch.load(os.path.join(self.model_save_path, 'checkpoint.pth')), strict=False)
        print('done')

        self.model.eval()
        with torch.no_grad():
            # test set
            attens_energy = []
            test_labels = []
            test_series = []
            for i, (batch_x, batch_y) in enumerate(test_loader):
                test_series.append(batch_x)
                test_labels.append(batch_y)

                batch_x = batch_x.float().to(self.device)
                score, q_latent_distance = self.model.infer(batch_x, None, None, None)
                score = score.detach().cpu().numpy()
                attens_energy.append(score)

            attens_energy = np.concatenate(attens_energy, axis=0)  # nb x t x c
            attens_energy = attens_energy.reshape(-1, attens_energy.shape[-1])  # nb*t x c
            test_energy = np.array(attens_energy)
            test_labels = np.concatenate(test_labels, axis=0).reshape(-1)  # nb*t
            test_labels = np.array(test_labels)
            test_gt = test_labels.astype(int)
            test_series = np.concatenate(test_series, axis=0)  # nb x t x c
            test_series = test_series.reshape(-1, test_series.shape[-1])  # nb*t x c
            test_series = np.array(test_series)

        np.save(self.rst_save_path + '/a_gt.npy', test_gt)  # nb*t
        np.save(self.rst_save_path + '/a_test_energy.npy', test_energy)  # nb*t x c

        test_energy = np.mean(test_energy, axis=-1)  # nb*t

        from ts_ad_evaluation import Evaluator
        evaluator = Evaluator(test_gt, test_energy, self.rst_save_path)
        results_storage = evaluator.evaluate(metrics=['accomplish_UCR'])
        evaluator.vis_anomaly_intervals_all(test_series)

        filename = os.path.join(self.args.save_path, self.args.data_origin, 'ucr.csv')
        results = [self.args.data, results_storage['accomplish_UCR']['topk'].values[0],
                   results_storage['accomplish_UCR']['total_len'].values[0],
                   results_storage['accomplish_UCR']['aplha_quantile'].values[0],
                   results_storage['accomplish_UCR']['3_alpha'].values[0],
                   results_storage['accomplish_UCR']['10_alpha'].values[0]]
        with open(filename, 'a') as f:
            f.write(','.join([str(result) for result in results]) + '\n')

    def analysis(self):
        self._basic()

class Configs:
    def __init__(self, json_path):
        with open(json_path) as f:
            configs = json.load(f)
            self.__dict__.update(configs)

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss
