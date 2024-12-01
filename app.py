from fastapi import FastAPI, UploadFile, File
from typing import Dict
import torch
import torchvision.transforms as transforms
import torchxrayvision as xrv
import skimage.io
import numpy as np
import io

app = FastAPI()

# Load model
weights = 'densenet121-res224-all'
resize = True
cuda = torch.cuda.is_available() #use GPU if it is avilable
model = xrv.models.get_model(weights)
if cuda:
    model = model.cuda()


# Define prediction function
def predict_image(image: np.ndarray) -> Dict[str, float]:
    # Normalize image
    image = xrv.datasets.normalize(image, 255)

    # Ensure image is 2D
    if len(image.shape) > 2:
        image = image[:, :, 0]
    if len(image.shape) < 2:
        raise ValueError("Image dimension is lower than 2")

    # Add color channel
    image = image[None, :, :]

    # Transform image
    transform = transforms.Compose([
        xrv.datasets.XRayCenterCrop(),
        xrv.datasets.XRayResizer(224) if resize else xrv.datasets.XRayCenterCrop()
    ])
    image = transform(image)

    # Convert to tensor and predict
    image_tensor = torch.from_numpy(image).unsqueeze(0)
    if cuda:
        image_tensor = image_tensor.cuda()

    with torch.no_grad():
        preds = model(image_tensor).cpu()
        # Convert numpy.float32 values to native Python floats
        output = {k: float(v) for k, v in zip(xrv.datasets.default_pathologies, preds[0].detach().numpy())}

    return output


# Define API endpoint
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Load image file
    image = skimage.io.imread(io.BytesIO(await file.read()))
    predictions = predict_image(image)
    return predictions

