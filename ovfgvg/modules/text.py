import clip
import lightning as L
from MinkowskiEngine import SparseTensor


class TextEncoderModule(L.LightningModule):
    EPSILON = 1e-5

    def __init__(self, model, **kwargs):
        super().__init__()
        self.model = model

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        text = batch
        # forward
        text_features = self.model.encode_text(text)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        return {"features": text_features.cpu()}
