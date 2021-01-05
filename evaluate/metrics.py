import torch.nn as nn
import torch.nn.functional as F
import torch

class DiceMetric(nn.Module):
    def __init__(self, smooth=1):
        super(DiceMetric, self).__init__()
        self.smooth = 1
        self.name = 'dice'
    
    def forward(self, inputs, targets):
        # flatten tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        denominator = inputs.sum() + targets.sum() + self.smooth

        dice = (2. * intersection + self.smooth) / denominator

        return dice 

class IoUMetric(nn.Module):
    def __init__(self, smooth=1):
        super(IoUMetric, self).__init__()
        self.smooth = smooth
        self.name = 'iou'
    
    def forward(self, inputs, targets):

        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        union = (inputs + targets).sum() - intersection
        iou = (intersection + self.smooth) / (union + self.smooth)

        return iou

class Recall(nn.Module):
    def __init__(self, smooth=1):
        super(Recall, self).__init__()
        self.smooth = 1
        self.name = 'recall'
        
    
    def forward(self, inputs, targets):

        inputs = inputs.view(-1)
        targets = targets.view(-1)

        TP = (inputs * targets).sum()
        FN = ((1 - inputs) * targets).sum()

        recall = (TP + self.smooth) / (TP + FN + self.smooth)

        return recall

class Precision(nn.Module):

    def __init__(self, smooth=1):
        super(Precision, self).__init__()
        self.smooth = smooth
        self.name = 'precision'
    
    def forward(self, inputs, targets):

        inputs = inputs.view(-1)
        targets = targets.view(-1)

        TP = (inputs * targets).sum()
        FP = ((1 - targets) * inputs).sum()

        precision = (TP + self.smooth) / (TP + FP + self.smooth)

        return torch.clip(precision, 0, 1)


class MetricsLogger(object):

    def __init__(self, metrics):

        self.metrics_values = dict()

        for m in metrics:
            if not isinstance(m, nn.Module):
                raise Exception('Metric must implements nn.Module')
            self.metrics_values[m.name] = []

        self.metrics = metrics
        self.iter_data = dict()

    def log(self, inputs, targets, seismic, location, iter_number):
        inputs = torch.tensor(inputs)
        targets = torch.tensor(targets)

        record = dict()

        record['true_mask'] = targets
        record['pred_mask'] = inputs
        record['seismic'] = seismic
        record['location'] = location

        for m in self.metrics:
            value = m.forward(inputs, targets)
            record[m.name] = value
            self.metrics_values[m.name].append(value)
        
        self.iter_data[iter_number] = record

    def iter_results(self):
        return self.iter_data
    
    def results(self):
        return self.metrics_values