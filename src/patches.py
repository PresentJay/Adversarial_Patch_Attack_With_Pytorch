
class AdversarialPatch():
    def __init__(self, image=None, mask=None, target=None):
        self.image = image
        self.mask = mask
        self.adv_image = None
        self.target = None
        self.shape = None
    
    def SetSquarePatch(self):
        pass
    
    def SetCirclePatch(self):
        pass