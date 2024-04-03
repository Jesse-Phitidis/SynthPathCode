import torch

class BaseInferer:
    
    @torch.no_grad()
    def __call__(self, model, image, anatomy):
        return model(image)