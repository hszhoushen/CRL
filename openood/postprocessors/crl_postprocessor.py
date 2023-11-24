from typing import Any
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_postprocessor import BasePostprocessor
from .info import num_classes_dict


class CRLPostprocessor(BasePostprocessor):
    def __init__(self, config):
        print("Class relevance learning framework")
        self.config = config
        self.num_classes = num_classes_dict[self.config.dataset.name]
        self.setup_flag = False
        self.args = self.config.postprocessor.postprocessor_args
        self.alpha = self.args.alpha
        print('self.alpha:', self.alpha)
        self.beta = self.args.beta
        print('self.beta:', self.beta)

        self.args_dict = self.config.postprocessor.postprocessor_sweep
        print('self.args_dict:', self.args_dict)

    def _get_relative_entropy(self, logits: torch.Tensor):
        relative_entr = []
        pred_labels = []  # Initialize an empty list to store Ipc values

        # Convert logits to PyTorch tensor and ensure no NaN values
        logits = torch.tensor(logits)
        logits = torch.nan_to_num(logits)

        for row in logits:
            Ipc = torch.argmax(row)
            # pk = F.softmax(row, dim=0) 
            # qk = F.softmax(self.prototype_logits[Ipc], dim=0)
            pk = F.softmax(row) 
            qk = F.softmax(self.prototype_logits[Ipc])
            
            # Pcr = torch.sum(pk * torch.log(pk / qk) )
            # Add a small constant for numerical stability
            epsilon = 1e-10

            # Calculate KL divergence
            Pcr = torch.sum(pk * torch.log((pk + epsilon) / (qk + epsilon)))

            relative_entr.append(Pcr.item())  # Convert to a scalar value and append
            pred_labels.append(Ipc.item())  # Append Ipc value to pred_labels list

        relative_entr = torch.tensor(relative_entr)
        pred_labels = torch.tensor(pred_labels)

        return relative_entr, pred_labels

    def _get_mixed_score(self, logits: torch.Tensor, alpha=1.0, beta=0.1):
        # Check the initial device of logits
        # print("logits Device Before:", logits.device)

        logits = logits.to('cuda:0')  # Ensure logits are on the GPU
        # Check the device after moving logits to GPU
        # print("logits Device After:", logits.device)

        relative_entr, pred_labels = self._get_relative_entropy(logits)
        # print("relative_entr Device:", relative_entr.device)
        relative_entr = relative_entr.to('cuda:0')
        
        alpha = torch.tensor(alpha).to('cuda:0')
        beta = torch.tensor(beta).to('cuda:0')
        # epsilon = torch.tensor(0.005).to('cuda:0')

        # Check the device of alpha, beta, and epsilon
        # print("alpha Device:", alpha.device)
        # print("beta Device:", beta.device)
        # print("epsilon Device:", epsilon.device)

        # Calculate pred_scores using PyTorch operations
        max_logits = torch.max(logits, dim=1)
        # Check the device of max_logits
        # print("max_logits Device:", max_logits.values.device)

        # pred_scores = -max_logits * alpha - 1 / (relative_entr + epsilon) * beta
        # pred_scores = max_logits.values.mul(-alpha).sub(1 / (relative_entr + epsilon).mul(beta))
        # pred_scores = max_logits.values.mul(alpha).add(1 / (relative_entr + epsilon).mul(beta))
        pred_scores = max_logits.values.mul(alpha).add(1 / (relative_entr).mul(beta))

        # pred_scores = - pred_scores

        # Check the device of pred_labels and pred_scores
        # print("pred_labels Device:", pred_labels.device)
        # print("pred_scores Device:", pred_scores.device)

        return pred_labels, pred_scores
        

    def _get_prototype_distribution(self):
        prototype_logits = []

        # Convert self.logits and self.labels to PyTorch tensors
        logits_tensor = torch.tensor(self.logits).to('cuda:0')
        labels_tensor = torch.tensor(self.labels).to('cuda:0')

        # Iterate over the classes
        for i in range(logits_tensor.size(1)):
            # Create a mask for the rows where labels match the current class 'i'
            mask = (labels_tensor == i).view(-1, 1)

            # Filter logits for the current class 'i'
            class_logits = torch.masked_select(logits_tensor, mask).view(-1, logits_tensor.size(1))

            # Calculate the mean logits for the current class 'i'
            class_mean_logits = class_logits.mean(dim=0)

            prototype_logits.append(class_mean_logits)

        # Stack the prototype logits into a tensor
        prototype_logits_tensor = torch.stack(prototype_logits)

        return prototype_logits_tensor

    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):
        if not self.setup_flag:
            self.logits, self.labels = self.sample_estimator(net, id_loader_dict['train'], self.num_classes)
            self.prototype_logits = self._get_prototype_distribution()
            print('self.prototype_logits:', self.prototype_logits.shape, self.prototype_logits)

            self.setup_flag = True
        else:
            pass

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        net.eval()
        # get predictions
        logits, feature_list = net(data, return_feature_list=True)
        pred_labels, pred_scores = self._get_mixed_score(logits, self.alpha, self.beta)

        return pred_labels, pred_scores

    def set_hyperparam(self, hyperparam: list):
        self.alpha = hyperparam[0]
        self.beta = hyperparam[1]

    def get_hyperparam(self):
        return [self.alpha, self.beta]

    @torch.no_grad()
    def sample_estimator(self, model, train_loader, num_classes):

        # Initialize empty lists to store the logits and labels
        all_logits = []
        all_labels = []

        model.eval()
        # collect features and compute gram metrix
        for batch in tqdm(train_loader, desc='loading the logits of training data'):

            data = batch['data'].cuda()
            # print('data.shape:', data.shape)
            label = batch['label']
            # print('label:', len(label), label)
            logits, feature_list = model(data, return_feature_list=True)
            # print('logits:', logits.shape)
            # label_list = tensor2list(label)

            # Append the logits and labels from the current batch to the respective lists
            all_logits.append(logits)
            all_labels.append(label)
        
        # Stack all the logits and labels tensors along the batch dimension
        stacked_logits = torch.cat(all_logits, dim=0)
        stacked_labels = torch.cat(all_labels, dim=0)
        # Now, 'stacked_logits' and 'stacked_labels' contain all the logits and labels stacked together
        print('stacked_logits shape:', stacked_logits.shape)
        print('stacked_labels shape:', stacked_labels.shape)

        return stacked_logits, stacked_labels
