# cnn_extractor.py
import torch
# torch.set_num_threads(1)
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import cv2

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])

model = None

def get_model():
    global model

    if model is None:
        model = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT
        )
        model.fc = torch.nn.Identity()
        model.eval()
        model.to("cpu")

    return model


def extract_cnn_features(image):

    model = get_model()

    img = Image.fromarray(
        cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    )

    img = transform(img).unsqueeze(0)

    with torch.no_grad():
        feat = model(img).cpu().numpy().flatten()

    return feat