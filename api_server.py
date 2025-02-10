# api_server.py
import argparse
import asyncio
import base64
import logging
import os
import tempfile
import traceback
import uuid
from io import BytesIO

import torch
import trimesh
import uvicorn
from PIL import Image
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from hy3dgen.rembg import BackgroundRemover
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline, FloaterRemover, DegenerateFaceRemover, FaceReducer
from hy3dgen.texgen import Hunyuan3DPaintPipeline
from hy3dgen.text2image import HunyuanDiTPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("controller")

SAVE_DIR = 'gradio_cache'
os.makedirs(SAVE_DIR, exist_ok=True)

def load_image_from_base64(image_b64):
    return Image.open(BytesIO(base64.b64decode(image_b64)))

class ModelWorker:
    def __init__(self, model_path='tencent/Hunyuan3D-2', device='cuda'):
        self.model_path = model_path
        self.device = device
        logger.info(f"Loading the model {model_path} on device {device}...")

        # Initialize pipelines
        self.rembg = BackgroundRemover()
        self.pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
            model_path,
            device=device
        )
        self.pipeline_fast = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
            model_path,
            subfolder='hunyuan3d-dit-v2-0-fast',
            variant='fp16'
        )
        self.pipeline_t2i = HunyuanDiTPipeline(
            'Tencent-Hunyuan/HunyuanDiT-v1.1-Diffusers-Distilled',
            device=device
        )
        self.pipeline_tex = Hunyuan3DPaintPipeline.from_pretrained(model_path)

    @torch.inference_mode()
    def generate(self, uid, params):
        fast_mode = params.get("fast", False)

        if 'image' in params:
            image_b64 = params["image"]
            image = load_image_from_base64(image_b64)
        else:
            if 'text' in params:
                text = params["text"]
                image = self.pipeline_t2i(text)
            else:
                raise ValueError("No input image or text provided")

        image = self.rembg(image)
        params['image'] = image

        if 'mesh' in params:
            mesh = trimesh.load(BytesIO(base64.b64decode(params["mesh"])), file_type='glb')
        else:
            seed = params.get("seed", 1234)
            params['generator'] = torch.Generator(self.device).manual_seed(seed)
            params['octree_resolution'] = params.get("octree_resolution", 256)
            params['num_inference_steps'] = params.get("num_inference_steps", 30)
            params['guidance_scale'] = params.get('guidance_scale', 7.5)
            params['mc_algo'] = 'mc'

            pipeline = self.pipeline_fast if fast_mode else self.pipeline
            mesh = pipeline(**params)[0]

        if params.get('texture', False):
            mesh = FloaterRemover()(mesh)
            mesh = DegenerateFaceRemover()(mesh)
            mesh = FaceReducer()(mesh, max_facenum=params.get('face_count', 40000))
            mesh = self.pipeline_tex(mesh, image)

        with tempfile.NamedTemporaryFile(suffix='.glb', delete=False) as temp_file:
            mesh.export(temp_file.name)
            temp_file.close()
            mesh = trimesh.load(temp_file.name)
            os.unlink(temp_file.name)
            save_path = os.path.join(SAVE_DIR, f'{str(uid)}.glb')
            mesh.export(save_path)

        torch.cuda.empty_cache()
        return save_path, uid


app = FastAPI()

# Serve static files (HTML, JS, CSS) from the "static" directory
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def serve_homepage():
    """ Serve the main HTML page """
    with open("static/index.html", "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)

@app.post("/generate")
async def generate(request: Request):
    params = await request.json()
    uid = uuid.uuid4()
    try:
        file_path, uid = worker.generate(uid, params)
        return FileResponse(file_path, media_type='model/gltf-binary')
    except ValueError:
        traceback.print_exc()
        return JSONResponse({"error": "Invalid input"}, status_code=400)
    except torch.cuda.CudaError:
        traceback.print_exc()
        return JSONResponse({"error": "CUDA error"}, status_code=500)
    except Exception:
        traceback.print_exc()
        return JSONResponse({"error": "Unexpected error"}, status_code=500)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--model_path", type=str, default='tencent/Hunyuan3D-2')
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    worker = ModelWorker(model_path=args.model_path, device=args.device)

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
