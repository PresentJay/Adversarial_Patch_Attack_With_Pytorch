
class Accuracy:
    def __init__(self):
        self.sum = 0
        self.count = 0
    
    
    def tic(self, val):
        self.sum += val
        self.count += 1
    
        
    def average(self):
        return self.sum / self.count
    

