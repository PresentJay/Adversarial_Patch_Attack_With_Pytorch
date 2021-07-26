import os

class log:
    def __init__(self, savedir, ext, mode='w'):
        self.file = open(os.path.join(savedir, 'log.' + ext), mode)
    
    def write(self, msg, end='\n', _print=False):
        self.file.write(msg + end)
        if _print:
            print(msg, end=end)
        
    def horizonLine(self, _print=False):
        self.write(msg='- - - - - - - - - - -', _print=_print)
        
    def save(self):
        self.file.close()