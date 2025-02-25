import torch.nn as nn
import torchvision.transforms as T
from transformers import CLIPVisionModel

class CLIPImageEmbedder(nn.Module):
    def __init__(self, version="openai/clip-vit-large-patch14", n_conds=10, is_trainable=True, *args, **kwargs):
        super().__init__()
        self.transformer = CLIPVisionModel.from_pretrained(version)
        self.n_conds = n_conds
        if not is_trainable:
            self.freeze()
        self.clip_trainsform = T.Compose([
            T.Resize((224,224)),
            T.Normalize((0.48145466, 0.4578275, 0.40821073),
                        (0.26862954, 0.26130258, 0.27577711))
        ])
    def freeze(self):
        self.transformer = self.transformer.eval()
        for p in self.parameters():
            p.requires_grad = False
    def forward(self, hint, *args, **kwargs):
        image = hint
        image = self.clip_trainsform((image+1)/2)
        outputs = self.transformer(
            pixel_values=image,
            output_hidden_states=True
        )

        conds = outputs.hidden_states  # length = 25
        conds = conds[-self.n_conds:]
        conds = [c[:,1:,:] for c in conds]  # [BS x 256 x 1024]
        return conds, None