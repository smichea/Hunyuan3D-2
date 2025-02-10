import argparse
import asyncio
import base64
import logging
import logging.handlers
import os
import sys
import tempfile
import threading
import traceback
import uuid
from io import BytesIO

import torch
import trimesh
import uvicorn
from PIL import Image
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, FileResponse

from hy3dgen.rembg import BackgroundRemover
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline, FloaterRemover, DegenerateFaceRemover, FaceReducer
from hy3dgen.texgen import Hunyuan3DPaintPipeline
from hy3dgen.text2image import HunyuanDiTPipeline

LOGDIR = '.'

server_error_msg = "**NETWORK ERROR DUE TO HIGH TRAFFIC. PLEASE REGENERATE OR REFRESH THIS PAGE.**"
moderation_msg = "YOUR INPUT VIOLATES OUR CONTENT MODERATION GUIDELINES. PLEASE TRY AGAIN."

SAVE_DIR = 'gradio_cache'
os.makedirs(SAVE_DIR, exist_ok=True)

worker_id = str(uuid.uuid4())[:6]

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("controller")


def load_image_from_base64(image):
    return Image.open(BytesIO(base64.b64decode(image)))


class ModelWorker:
    def __init__(self, model_path='tencent/Hunyuan3D-2', device='cuda'):
        self.model_path = model_path
        self.worker_id = worker_id
        self.device = device
        logger.info(f"Loading the model {model_path} on worker {worker_id} ...")

        self.rembg = BackgroundRemover()
        self.pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(model_path, device=device)
        self.pipeline_fast = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
            model_path,
            subfolder='hunyuan3d-dit-v2-0-fast',
            variant='fp16'
        )
        self.pipeline_t2i = HunyuanDiTPipeline('Tencent-Hunyuan/HunyuanDiT-v1.1-Diffusers-Distilled',
                                               device=device)
        self.pipeline_tex = Hunyuan3DPaintPipeline.from_pretrained(model_path)

    @torch.inference_mode()
    def generate(self, uid, params):
        fast_mode = params.get("fast", False)
        
        if 'image' in params:
            image = params["image"]
            image = load_image_from_base64(image)
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
            mesh = trimesh.load(temp_file.name)
            temp_file.close()
            os.unlink(temp_file.name)
            save_path = os.path.join(SAVE_DIR, f'{str(uid)}.glb')
            mesh.export(save_path)

        torch.cuda.empty_cache()
        return save_path, uid


app = FastAPI()

@app.post("/generate")
async def generate(request: Request):
    logger.info("Worker generating...")
    params = await request.json()
    uid = uuid.uuid4()
    try:
        file_path, uid = worker.generate(uid, params)
        return FileResponse(file_path)
    except ValueError as e:
        traceback.print_exc()
        ret = {"text": server_error_msg, "error_code": 1}
        return JSONResponse(ret, status_code=404)
    except torch.cuda.CudaError as e:
        traceback.print_exc()
        ret = {"text": server_error_msg, "error_code": 1}
        return JSONResponse(ret, status_code=404)
    except Exception as e:
        traceback.print_exc()
        ret = {"text": server_error_msg, "error_code": 1}
        return JSONResponse(ret, status_code=404)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8081)
    parser.add_argument("--model_path", type=str, default='tencent/Hunyuan3D-2')
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--limit-model-concurrency", type=int, default=5)
    args = parser.parse_args()

    model_semaphore = asyncio.Semaphore(args.limit_model_concurrency)
    worker = ModelWorker(model_path=args.model_path, device=args.device)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
