import os

class log:
    def __init__(self, savedir, ext):
        self.file = open(os.path.join(savedir, 'log.' + ext), 'w')
    
    def write(self, msg, end='\n', _print=False):
        self.file.write(msg + end)
        if _print:
            print(msg)
        
    def horizonLine(self, _print=False):
        self.write(msg='- - - - - - - - - - -', _print=_print)
        
    def save(self):
        self.file.close()