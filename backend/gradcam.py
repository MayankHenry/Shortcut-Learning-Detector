import io
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms

class GradCAM:
    def __init__(self, model, target_layer=None, use_cuda=False):
        self.model = model
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")
        self.model.to(self.device)

        # Find target layer (last Conv2d) if not provided
        if target_layer is None:
            self.target_layer = self._find_last_conv_layer()
        else:
            self.target_layer = target_layer

        self.activations = None
        self.gradients = None

        # register forward hook to capture activations
        def forward_hook(module, input, output):
            self.activations = output
            # register gradient hook on the activations tensor
            def _save_grad(grad):
                self.gradients = grad
            try:
                output.register_hook(_save_grad)
            except Exception:
                # some outputs may be tuples
                pass

        self.target_layer.register_forward_hook(forward_hook)

        # preprocessing transformation (ImageNet)
        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def _find_last_conv_layer(self):
        import torch.nn as nn
        target = None
        for name, module in reversed(list(self.model.named_modules())):
            if isinstance(module, nn.Conv2d):
                target = module
                break
        if target is None:
            raise RuntimeError("No Conv2d layer found in model")
        return target

    def generate(self, pil_img, target_class=None):
        # keep original size
        orig_w, orig_h = pil_img.size

        img_t = self.preprocess(pil_img).unsqueeze(0).to(self.device)

        img_t.requires_grad = True

        logits = self.model(img_t)
        probs = F.softmax(logits, dim=1)
        pred_idx = int(probs.argmax(dim=1).item())
        pred_conf = float(probs[0, pred_idx].item())

        if target_class is None:
            target_class = pred_idx

        # backward on the target class score
        self.model.zero_grad()
        score = logits[0, target_class]
        score.backward(retain_graph=True)

        # get saved activations and gradients
        if self.activations is None or self.gradients is None:
            raise RuntimeError("Failed to obtain activations or gradients for Grad-CAM")

        activations = self.activations.detach().cpu()[0]  # C x H x W
        gradients = self.gradients.detach().cpu()[0]  # C x H x W

        # global average pooling of gradients
        weights = torch.mean(gradients, dim=(1, 2))  # C

        # weighted combination
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]

        cam = F.relu(cam)
        cam_np = cam.numpy()

        # normalize cam
        cam_np -= cam_np.min()
        if cam_np.max() != 0:
            cam_np /= cam_np.max()

        # convert to 0-255 and apply colormap (using matplotlib)
        import matplotlib.cm as cm

        colormap = cm.get_cmap("jet")
        heatmap = (colormap(cam_np)[:, :, :3] * 255).astype(np.uint8)

        # convert to PIL image and resize to original image size
        heatmap_img = Image.fromarray(heatmap)
        heatmap_img = heatmap_img.resize((orig_w, orig_h), resample=Image.BICUBIC)

        return pred_idx, pred_conf, heatmap_img
