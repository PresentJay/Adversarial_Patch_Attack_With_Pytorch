import torch

class Accuracy:
    def __init__(self):
        self.sum = 0
        self.count = 0
    
    
    def tic(self, val, size):
        self.sum += val
        self.count += size
        
    def show_status(self):
        print(self.sum, ':', self.count)
    
        
    def average(self):
        return self.sum / self.count
    

    def calculate(self, model_output, label):
        with torch.no_grad():
            prob, pred = model_output.topk(max((1, )), 1, True, True)
            pred = pred.t()
            correct = pred.eq(label.view(1, -1).expand_as(pred))
            results = []
            for n in (1, ):
                corrects = correct[:n].view(-1).float().sum(0, keepdims=True)
                results.append(corrects.mul_(100.0 / model_output.size(0)))
            self.tic(results[0].item(), model_output.size(0))
            self.show_status()
            return results