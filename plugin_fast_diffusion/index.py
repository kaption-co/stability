from utils.plugin import Plugin


class Fast_Diffusion(Plugin):
    def __init__(self, parent):
        super(Fast_Diffusion, self).__init__(parent)
        self.parent = parent
        self.name = "Fast Diffusion"
        self.category = "Diffusion"

    def train(self):
        pass

    def infer(self):
        pass
