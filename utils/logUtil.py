import os

class log:
    def __init__(self, savedir, ext, mode='w', name='log'):
        self.path = os.path.join(savedir,  name + '.' + ext)
        self.file = open(self.path, mode)
    
    def write(self, msg, end='\n', _print=False):
        self.file.write(msg + end)
        if _print:
            print(msg, end=end)
        
    def horizonLine(self, _print=False):
        self.write(msg='- - - - - - - - - - -', _print=_print)
        
    def save(self):
        self.file.close()