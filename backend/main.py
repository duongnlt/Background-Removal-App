from fastapi import FastAPI
from fastapi import FastAPI, File, UploadFile, Response
from typing import Union
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from inference import infer
from io import BytesIO
from PIL import Image
import cv2
from utils import make_warp_img

import numpy as np
app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Hello World"} 



@app.post("/predict", status_code=201, responses={
        200: {
            "content": {"image/png": {}}
        }
    }
)
async def predict(image: UploadFile):
    print(image.file)
    if not image:
        return {"message": "No image sent"}
    else:
        img = Image.open(BytesIO(await image.read()))
        img_np = np.array(img)
        result = infer(img_np)
        mask_img = np.array(result)
        print(type(mask_img))
        img_np[~mask_img.astype(bool)] = 0.
        warped_img = make_warp_img(img_np, mask_img)
        warped_img = cv2.cvtColor(warped_img, cv2.COLOR_BGR2RGB)

        encoded_png = cv2.imencode(".jpg", warped_img)[1]
        return Response(content=encoded_png.tobytes(), media_type="image/jpg")
    

if __name__ == '__main__':
    uvicorn.run(app= 'main:app', host='0.0.0.0', port=8000, reload=True)