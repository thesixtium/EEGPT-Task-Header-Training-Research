import torch.nn as nn

class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=(64, 1000), patch_size=16, patch_stride=None, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        if patch_stride is None:
            self.num_patches = ((img_size[0]), (img_size[1] // patch_size))
        else:
            self.num_patches = ((img_size[0]), ((img_size[1] - patch_size) // patch_stride + 1))

        self.proj = nn.Conv2d(1, embed_dim, kernel_size=(1, patch_size),
                              stride=(1, patch_size if patch_stride is None else patch_stride))

    def forward(self, x):
        # x: B,C,T
        x = x.unsqueeze(1)  # B, 1, C, T
        x = self.proj(x).transpose(1, 3)  # B, T, C, D
        return x