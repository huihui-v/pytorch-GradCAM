"""
Simple PyTorch implementation of Grad-CAM.
"""

# Imports
import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models
from torchvision import transforms as T


# Helper functions

def load_img(imgpath):
    """Load image.
    
    Args:
        imgpath (string): The path of the image to load.

    Returns:
        ((int, int), torch.Tensor): The size of original image, and the normalized image tensor. 

    """
    img = Image.open(imgpath)
    ori_size = img.size
    
    transforms = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img_tensor = transforms(img)

    return (ori_size, img_tensor)

def save_img(img_tensor, grad_cam, path):
    """Save result image.

    This function will stack the Grad-CAM mask onto the original image, then write to file.

    Args:
        img_tensor (torch.Tensor): The image tensor used by the model, shape (3, 224, 224).
        grad_cam (torch.Tensor): The Grad-CAM output, shape (1, H, W), where H and W denotes the height and width of feature map.
        path (string): The output file path.
    
    Returns:
        None.

    """

    grad_cam -= grad_cam.min()
    grad_cam /= grad_cam.max()
    mask = grad_cam.detach().cpu().numpy()

    heatmap = cv2.applyColorMap(np.uint8(255*mask), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = cv2.resize(heatmap, (img_tensor.shape[1], img_tensor.shape[2]))

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    for i in range(3):
        img_tensor[i] *= std[i]
        img_tensor[i] += mean[i]

    img = img_tensor.detach().cpu().permute(1, 2, 0).numpy()
    img = img + np.float32(heatmap)/255
    img /= img.max()
    img = np.uint8(255*img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    cv2.imwrite(f"{path}", img)
    print(f"Save {path} complete")

# Model

class GradCAM(nn.Module):
    """Grad-CAM class.

    The whole procedure of the Grad-CAM method are stored in this class. You can change the model to show as long as it can be splited into
    `extractor` part and `further` part.

    """
    def __init__(self):
        """Initial

        You should change the code here to use other models.

        Args:
            None.

        Returns:
            None.

        """
        super(GradCAM, self).__init__()
        self.bone = models.vgg16(pretrained=True)
        self.extractor = self.bone.features
        self.downstream = nn.Sequential()
        self.downstream.add_module(name='avgpool', module=self.bone.avgpool)
        self.downstream.add_module(name='classifier', module=self.bone.classifier)
        
        for param in self.extractor.parameters():
            param.requires_grad = False

        for param in self.downstream.parameters():
            param.requires_grad = True

    def forward(self, x, class_idx):
        """Forward function.

        Args:
            x (torch.Tensor): The input image tensor, shape (C, H, W)
            class_idx (int): The index of the class in Imagenet.
        
        Return:
            (torch.Tensor) grad_cam, the Grad-CAM mask, shape (1, H', W'), where H' and W' are the height and width of the feature map.

        """

        # Make it as a batch
        x = x.unsqueeze(0)
        
        # Forward
        A = self.extractor(x)
        A.requires_grad = True
        result = self.downstream.avgpool(A)
        result = result.view(1, -1)
        result = self.downstream.classifier(result)
        
        # Compute gradient
        result[0, class_idx].backward()

        # Compute alpha
        alpha = A.grad.mean(dim=[2, 3], keepdim=True)

        # Compute Grad-CAM
        weighted_sum = torch.sum(alpha * A, dim=[0, 1])
        grad_cam = nn.functional.relu(weighted_sum)

        return grad_cam

def main():
    device = torch.device('cuda:0')
    model = GradCAM()

    _, img_tensor = load_img('demo/0.png')
    img_tensor = img_tensor.to(device)
    model = model.to(device)

    class_idx = 262
    grad_cam = model(img_tensor, class_idx)

    save_img(img_tensor, grad_cam, f"demo/0-{class_idx}.png")

main()
