import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm import tqdm
import wandb
import numpy as np


PATTERN = np.array([
    [1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1],
    [1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1],
    [1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1],
    [0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1],
    [0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1],
    [0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1]
])

class SemiConnectedConv(nn.Module):
    def __init__(self, config, pattern):
        super(SemiConnectedConv, self).__init__()
        self.pattern = pattern
        self.layers = []
        convs_list = [nn.Conv2d(in_channels=pattern[:, i].sum(), out_channels=1, kernel_size=config['kernel_size'], stride=config['stride'], padding=config['padding']) for i in range(pattern.shape[1])]
        convs_list.append(nn.Sigmoid())
        self.layers = nn.ModuleList(convs_list)
    

    def forward(self, x):
        x_c = []
        for i in range(self.pattern.shape[1]):
            indices = np.where(self.pattern[:, i] == 1)[0]
            x_i = x[:,indices, :, :]
            x_i = self.layers[i](x_i)
            x_c.append(x_i)
        x_c = torch.cat(x_c, dim=1)
        x_c = self.layers[-1](x_c)
        return x_c


class Lenet(nn.Module):
    def __init__(self, config):
        super(Lenet, self).__init__()
        self.config = config
        self.layers = []
        for layer in config['architecture']:
            if layer['type'] == 'conv':
                self.layers.append(nn.Conv2d(layer['in_channels'], layer['out_channels'], layer['kernel_size'], layer['stride'], layer['padding']))
                # normalization first
                if 'norm' in layer and layer['norm'] == 'batchnorm':
                    self.layers.append(nn.BatchNorm2d(layer['out_channels']))
                
                # next activation function
                if layer['activation'] == 'sigmoid':
                    self.layers.append(nn.Sigmoid())
                if layer['activation'] == 'relu':
                    self.layers.append(nn.ReLU())
                if layer['activation'] == 'leaky_relu':
                    self.layers.append(nn.LeakyReLU(self.config['leaky_relu_slope']))
                
                # dropout last
                if 'dropout' in layer and layer['dropout'] > 0.0:
                    self.layers.append(nn.Dropout2d(layer['dropout']))
                

            elif layer['type'] == 'avg_pool':
                self.layers.append(nn.AvgPool2d(layer['kernel_size'], layer['stride']))

            elif layer['type'] == 'semi_connected_conv':
                # implementation of original paper semi connected conv layer where each kernel is applied to a subset of the input channels
                pattern = PATTERN
                self.layers.append(SemiConnectedConv(config = layer, pattern=pattern))

                if 'norm' in layer and layer['norm'] == 'batchnorm':
                    self.layers.append(nn.BatchNorm2d(layer['out_channels']))
                
                if layer['activation'] == 'sigmoid':
                    self.layers.append(nn.Sigmoid())
                if layer['activation'] == 'relu':
                    self.layers.append(nn.ReLU())
                if layer['activation'] == 'leaky_relu':
                    self.layers.append(nn.LeakyReLU(self.config['leaky_relu_slope']))
                
                if 'dropout' in layer and layer['dropout'] > 0.0:
                    self.layers.append(nn.Dropout2d(layer['dropout']))

            elif layer['type'] == 'fc':
                self.layers.append(nn.Linear(layer['in_features'], layer['out_features']))

                # normalization first
                if 'norm' in layer and layer['norm'] == 'batchnorm':
                    self.layers.append(nn.BatchNorm1d(layer['out_features']))
                
                # next activation function
                if layer['activation'] == 'sigmoid':
                    self.layers.append(nn.Sigmoid())
                if layer['activation'] == 'relu':
                    self.layers.append(nn.ReLU())
                if layer['activation'] == 'leaky_relu':
                    self.layers.append(nn.LeakyReLU(self.config['leaky_relu_slope']))
                if layer['activation'] == 'softmax':
                    self.layers.append(nn.Softmax())
                
                # dropout last
                if 'dropout' in layer and layer['dropout'] > 0.0:
                    self.layers.append(nn.Dropout(layer['dropout']))
            
            elif layer['type'] == 'flatten':
                self.layers.append(nn.Flatten())
            
            elif layer['type'] == 'dropout1d':
                self.layers.append(nn.Dropout(layer['p']))
            
            elif layer['type'] == 'dropout2d':
                self.layers.append(nn.Dropout2d(layer['p']))
            
            elif layer['type'] == 'batch_norm2d':
                self.layers.append(nn.BatchNorm2d(layer['num_features']))
            
            elif layer['type'] == 'batch_norm1d':
                self.layers.append(nn.BatchNorm1d(layer['num_features']))
        
        self.layers = nn.ModuleList(self.layers)
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class CNNClassifier(nn.Module):
    def __init__(self, config):
        super(CNNClassifier, self).__init__()
        self.config = config
        self.model = Lenet(config)

        if self.config['kaiming_init']:
            self.model.apply(self.init_weights)
        
        self.criterion = nn.CrossEntropyLoss()
        self.learning_rate = config['learning_rate']
        self.batch_size = config['batch_size']
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        if self.config['scheduler'] == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=self.config['step_size'], gamma=self.config['gamma'])
        
        self.val_progress = []
        self.train_progress = []
        self.test_progress = []
        self.val_loss = 0.0
        self.train_loss = 0.0
        self.best_acc = 0
        self.best_f1 = 0
        self.best_epoch = 0
        self.current_epoch = 0

    def forward(self, x):
        return self.model(x)
    

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, nonlinearity=self.config['init_nonlinearity'])
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    

    def fit(self, train_loader, val_loader):
        for epoch in range(self.config['epochs']):
            self.current_epoch = epoch
            print(f'Epoch {epoch}')
            self.train()
            print('Train')
            for batch in tqdm(train_loader):
                self.optimizer.zero_grad()
                loss = self.training_step(batch, 0)
                loss.backward()
                self.optimizer.step()
            if self.config['scheduler'] == 'step':
                self.scheduler.step()
            self.eval()
            print('Validation')
            for batch in tqdm(val_loader):
                with torch.no_grad():
                    loss = self.validation_step(batch, 0)
            self.on_validation_epoch_end()
        return 
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        probs = self(x)
        loss = self.criterion(probs, y)
        self.train_loss += loss.item()
        self.train_progress.append((probs, y))
        return loss

    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        probs = self(x)
        loss = self.criterion(probs, y)
        self.val_loss += loss.item()
        # save predictions and true labels to calculate accuracy on full validation after epoch end
        self.val_progress.append((probs, y))
        return loss
    
    def on_validation_epoch_end(self):
        # calculate accuracy on full validation set
        preds = torch.cat([pred for pred, y in self.val_progress], dim=0)
        y = torch.cat([y for pred, y in self.val_progress], dim=0)
        preds = torch.argmax(preds, dim=1)
        acc = accuracy_score(y.cpu(), preds.cpu())
        precision, recall, f1, _ = precision_recall_fscore_support(y.cpu(), preds.cpu(), average='macro')

        if len(self.train_progress) > 0:
            train_preds = torch.cat([pred for pred, y in self.train_progress], dim=0)
            train_y = torch.cat([y for pred, y in self.train_progress], dim=0)
            train_preds = torch.argmax(train_preds, dim=1)
            train_acc = accuracy_score(train_y.cpu(), train_preds.cpu())
        else:
            train_acc = 0.0

        # log metrics based on configuration settings
        if self.config['log'] in ['wandb', 'all']:
            log_dict = {'train_loss': self.train_loss, 'train acc': train_acc, 'val_loss': self.val_loss, 'val_acc': acc, 'val_precision': precision, 'val_recall': recall, 'val_f1': f1}
            if self.config['scheduler'] == 'step':
                log_dict['lr'] = self.scheduler.get_last_lr()[0]
            wandb.log(log_dict)
        if self.config['log'] in ['stdout', 'all']:
            print(f'Training loss: {self.train_loss}')
            print(f'Validation accuracy: {acc}')
            print(f'Validation loss: {self.val_loss}')
            print(f'Validation precision: {precision}')
            print(f'Validation recall: {recall}')
            print(f'Validation f1: {f1}')
            if self.config['scheduler'] == 'step':
                print(f'Learning rate: {self.scheduler.get_last_lr()[0]}')
        
        # saved best model, based on chosen target metric if save_best is set to True
        if self.config['save_best']:
            if self.config['target_metric'] == 'accuracy' and acc >= self.best_acc:
                self.best_acc = acc
                self.save_checkpoint('weights/'+self.config['run_name']+'.pth')
                self.best_epoch = self.current_epoch
                self.best_f1 = f1
            elif self.config['target_metric'] == 'f1' and f1 >= self.best_f1:
                self.best_f1 = f1
                self.save_checkpoint('weights/'+self.config['run_name']+'.pth')
                self.best_epoch = self.current_epoch
                self.best_acc = acc

        # reset variables for next epoch
        self.val_progress = []
        self.val_loss = 0.0
        self.train_loss = 0.0
        self.train_progress = []
    

    def test_step(self, batch, batch_idx):
        """
        Test step for MEGConvNet.

        :param batch: Batch of training data.
        :param batch_idx: Index of the current batch.
        :returns: loss: The test loss.
        """
        x, y = batch
        probs = self(x)
        loss = self.criterion(probs, y)
        self.test_progress.append((probs, y))
        return loss
    
    def on_test_epoch_end(self):
        """
        Test callback for pl trainer.
        Calculates and logs test metrics.
        """
        # calculate accuracy on full test set
        preds = torch.cat([pred for pred, y in self.test_progress], dim=0)
        y = torch.cat([y for pred, y in self.test_progress], dim=0)
        preds = torch.argmax(preds, dim=1)
        acc = accuracy_score(y.cpu(), preds.cpu())
        precision, recall, f1, _ = precision_recall_fscore_support(y.cpu(), preds.cpu(), average='macro')

        # log metrics based on configuration settings
        if self.config['log'] in ['wandb', 'all']:
            wandb.log({'test_acc': acc, 'test_precision': precision, 'test_recall': recall, 'test_f1': f1})
        if self.config['log'] in ['stdout', 'all']:
            print(f'Test accuracy: {acc}')
            print(f'Test precision: {precision}')
            print(f'Test recall: {recall}')
            print(f'Test f1: {f1}')
        
        # reset variables for next epoch
        self.test_progress = []
    

    def save_checkpoint(self, path):
        """
        Saves a checkpoint, optimizer, and scheduler from a specified path.
        :param path: The file path from which to load the checkpoint.
        """
        cnn_state = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }
        if self.config['scheduler'] is not None:
            cnn_state['scheduler_state_dict'] = self.scheduler.state_dict()
        torch.save(cnn_state, path)

    def load_checkpoint(self, path):
        """
        Loads a checkpoint, optimizer, and scheduler from a specified path.
        :param path: The file path from which to load the checkpoint.
        """
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.config['scheduler'] is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])