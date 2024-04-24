from torch import nn
from model.transformers import TransformerEncoder, ClassificationHead
from model.convolution import PatchEmbedding


class Conformer(nn.Sequential):
    def __init__(self, emb_size=40, nb_channels =23, depth=6, n_classes=2, **kwargs):
        super().__init__(
            PatchEmbedding(emb_size, nb_channels),
            TransformerEncoder(depth, emb_size),
            ClassificationHead(emb_size, n_classes)
        )
