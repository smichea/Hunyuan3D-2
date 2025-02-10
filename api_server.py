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
import gc
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
SAVE_DIR = 'gradio_cache'
os.makedirs(SAVE_DIR, exist_ok=True)
worker_id = str(uuid.uuid4())[:6]

server_error_msg = "**NETWORK ERROR DUE TO HIGH TRAFFIC. PLEASE REGENERATE OR REFRESH THIS PAGE.**"

# Logging setup
def build_logger(logger_name, logger_filename):
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(logger_filename, encoding='UTF-8')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

logger = build_logger("controller", f"{SAVE_DIR}/controller.log")

def load_image_from_base64(image):
    return Image.open(BytesIO(base64.b64decode(image)))

class ModelWorker:
    def __init__(self, model_path='tencent/Hunyuan3D-2', device='cuda'):
        self.model_path = model_path
        self.worker_id = worker_id
        self.device = device
        logger.info(f"Loading the model {model_path} on worker {worker_id} ...")

        self.rembg = BackgroundRemover()
        self.pipeline_t2i = HunyuanDiTPipeline('Tencent-Hunyuan/HunyuanDiT-v1.1-Diffusers-Distilled', device=device)
        self.pipeline_tex = Hunyuan3DPaintPipeline.from_pretrained(model_path)

    @torch.inference_mode()
    def generate(self, uid, params):
        try:
            fast_mode = params.get("fast", False)
            
            if 'image' in params:
                image = load_image_from_base64(params["image"])
            elif 'text' in params:
                text = params["text"]
                image = self.pipeline_t2i(text)
            else:
                raise ValueError("No input image or text provided")

            image = self.rembg(image)

            if fast_mode:
                pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
                    self.model_path,
                    subfolder='hunyuan3d-dit-v2-0-fast',
                    variant='fp16',
                    device=self.device
                )
            else:
                pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(self.model_path, device=self.device)

            params['generator'] = torch.Generator(self.device).manual_seed(params.get("seed", 1234))
            params['num_inference_steps'] = params.get("num_inference_steps", 20)
            params['guidance_scale'] = params.get('guidance_scale', 5.0)
            params['mc_algo'] = 'mc'
            
            mesh = pipeline(**params)[0]
            
            if params.get('texture', False):
                mesh = FloaterRemover()(mesh)
                mesh = DegenerateFaceRemover()(mesh)
                mesh = FaceReducer()(mesh, max_facenum=params.get('face_count', 40000))
                mesh = self.pipeline_tex(mesh, image)

            save_path = os.path.join(SAVE_DIR, f'{str(uid)}.glb')
            mesh.export(save_path)

            return save_path, uid
        
        except Exception as e:
            logger.error(f"Error during generation: {e}")
            traceback.print_exc()
            raise e
        
        finally:
            del pipeline, mesh, image
            torch.cuda.empty_cache()
            gc.collect()

app = FastAPI()

@app.post("/generate")
async def generate(request: Request):
    logger.info("Worker generating...")
    params = await request.json()
    logger.info(f"Request params: {params}")
    uid = uuid.uuid4()
    try:
        file_path, uid = worker.generate(uid, params)
        return FileResponse(file_path)
    except ValueError as e:
        logger.error(f"ValueError: {e}")
        traceback.print_exc()
        return JSONResponse({"text": server_error_msg, "error_code": 1}, status_code=404)
    except Exception as e:
        logger.error(f"Unknown Error: {e}")
        traceback.print_exc()
        return JSONResponse({"text": server_error_msg, "error_code": 1}, status_code=404)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--model_path", type=str, default='tencent/Hunyuan3D-2')
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    
    worker = ModelWorker(model_path=args.model_path, device=args.device)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
