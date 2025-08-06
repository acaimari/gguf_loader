# cinewan/modules/wan_transformer_gguf.py
# GGUF-optimized version of WanTransformer3DModel
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# This file is a modified version of:
#   https://github.com/aigc-apps/VideoX-Fun/blob/main/examples/wan2.1/predict_t2v.py
# which was originally based on:
#   https://github.com/Wan-Video/Wan2.1/
#
# GGUF support and further modifications by https://github.com/acaimari
#
# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
"""
Script to run GGUF T2V inference with intelligent memory management
"""

import os
import sys
import torch
import logging
import argparse
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from cinewan.gguf_loader import load_gguf_pipeline
import imageio
import numpy as np

# --- NUEVAS IMPORTACIONES ---
from videox_fun.utils.utils import filter_kwargs
from videox_fun.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from videox_fun.utils.fm_solvers import FlowDPMSolverMultistepScheduler
from diffusers import FlowMatchEulerDiscreteScheduler
# --- FIN DE NUEVAS IMPORTACIONES ---

# Configuración del logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def print_gpu_memory():
    """Imprime el uso actual de la memoria GPU"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        logger.info(f"Memoria GPU: {allocated:.2f}GB en uso, {reserved:.2f}GB reservados")


def main():
    parser = argparse.ArgumentParser(description='Run GGUF T2V inference with smart memory management')
    parser.add_argument('--gguf_path', type=str, required=True,
                        help='Ruta al archivo GGUF')
    parser.add_argument('--base_model_path', type=str, required=True,
                        help='Ruta a los componentes del modelo base (VAE, text encoder)')
    parser.add_argument('--config_path', type=str, required=True,
                        help='Ruta al archivo de configuración YAML del modelo')
    parser.add_argument('--prompt', type=str, default="A beautiful sunset over mountains",
                        help='Prompt de texto para la generación')
    parser.add_argument('--negative_prompt', type=str, default="",
                        help='Prompt negativo')
    parser.add_argument('--height', type=int, default=320,
                        help='Altura del vídeo')
    parser.add_argument('--width', type=int, default=512,
                        help='Anchura del vídeo')
    parser.add_argument('--video_length', type=int, default=17,
                        help='Número de frames')
    parser.add_argument('--num_inference_steps', type=int, default=30,
                        help='Número de pasos de inferencia')
    parser.add_argument('--guidance_scale', type=float, default=7.5,
                        help='Guidance scale')
    parser.add_argument('--seed', type=int, default=42,
                        help='Seed aleatoria')
    parser.add_argument('--output_path', type=str, default='output_gguf.mp4',
                        help='Ruta de salida del vídeo')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Dispositivo a usar')
    parser.add_argument('--dtype', type=str, default='bfloat16',
                        help='Tipo de peso (float32, float16, bfloat16)')
    parser.add_argument('--smart_memory', action='store_true', default=True,
                        help='Uso de gestión inteligente de memoria (por defecto: True)')
    parser.add_argument('--enable_teacache', action='store_true',
                        help='Activar aceleración TeaCache')
    parser.add_argument('--enable_vae_tiling', action='store_true', default=True,
                        help='Activar VAE tiling para ahorrar memoria')

    # --- NUEVOS ARGUMENTOS ---
    parser.add_argument("--sampler_name", type=str, default="Flow_Unipc",
                        choices=["Flow", "Flow_Unipc", "Flow_DPM++"],
                        help="Sampler a utilizar. Flow_Unipc es el recomendado para modelos Wan.")
    parser.add_argument("--shift", type=float, default=3.0,
                        help="Parámetro de shift para el scheduler en algunos samplers.")
    # --- FIN DE NUEVOS ARGUMENTOS ---

    args = parser.parse_args()

    # Set dtype
    dtype_map = {
        'float32': torch.float32,
        'float16': torch.float16,
        'bfloat16': torch.bfloat16
    }
    weight_dtype = dtype_map.get(args.dtype, torch.bfloat16)

    logger.info("="*60)
    logger.info("🚀 GGUF T2V Inference with Smart Memory Management")
    logger.info("="*60)
    logger.info(f"Dispositivo: {args.device}")
    logger.info(f"Tipo de peso: {weight_dtype}")
    logger.info(f"Archivo GGUF: {args.gguf_path}")
    logger.info(f"Modelo base: {args.base_model_path}")
    logger.info(f"Smart memory: {args.smart_memory}")
    logger.info("="*60)

    # Comprobar memoria inicial
    print_gpu_memory()

    # Cargar configuración
    from omegaconf import OmegaConf
    config = OmegaConf.load(args.config_path)
    logger.info("✅ Configuración cargada")

    # --- LÓGICA DE CREACIÓN DEL SCHEDULER CORRECTO ---
    scheduler_map = {
        "Flow": FlowMatchEulerDiscreteScheduler,
        "Flow_Unipc": FlowUniPCMultistepScheduler,
        "Flow_DPM++": FlowDPMSolverMultistepScheduler,
    }
    SchedulerClass = scheduler_map[args.sampler_name]
    
    # Leer los parámetros del scheduler desde el archivo de configuración
    scheduler_kwargs = config.get('scheduler_kwargs', {})
    scheduler_kwargs = OmegaConf.to_container(scheduler_kwargs)  # Convertir a dict Python
    
    # Algunos schedulers requieren un 'shift' específico
    if args.sampler_name in ["Flow_Unipc", "Flow_DPM++"]:
        scheduler_kwargs['shift'] = 1
        
    scheduler = SchedulerClass(**filter_kwargs(SchedulerClass, scheduler_kwargs))
    logger.info(f"✅ Scheduler '{args.sampler_name}' creado exitosamente")
    # --- FIN DE LÓGICA DE CREACIÓN ---

    # Cargar GGUF pipeline
    logger.info("\n📋 Cargando pipeline GGUF...")
    print_gpu_memory()

    pipeline = load_gguf_pipeline(
        gguf_path=args.gguf_path,
        base_model_path=args.base_model_path,
        config_path=args.config_path,
        device='cpu',  # Cargar primero en CPU
        dtype=weight_dtype,
        scheduler=scheduler,
    )

    logger.info("✅ ¡Pipeline cargado con éxito!")
    print_gpu_memory()

    # Smart memory management
    if args.smart_memory:
        logger.info("\n🧠 Usando gestión inteligente de memoria")
        logger.info("Moviendo Transformer y Text Encoder a GPU...")
        pipeline.transformer = pipeline.transformer.to(args.device)
        pipeline.text_encoder = pipeline.text_encoder.to(args.device)
        # VAE se queda en CPU de momento
        pipeline.vae = pipeline.vae.to('cpu')
        logger.info("✅ Transformer en GPU, VAE en CPU")
    else:
        # Todo en GPU
        logger.info("\n📦 Moviendo todo el pipeline a GPU...")
        pipeline = pipeline.to(args.device)

    print_gpu_memory()

    # Activar TeaCache si se solicita
    if args.enable_teacache:
        logger.info("\n☕ Activando TeaCache...")
        try:
            pipeline.transformer.enable_teacache(
                threshold=0.1,
                device=args.device,
                dtype=weight_dtype
            )
            logger.info("✅ TeaCache activado")
        except Exception as e:
            logger.warning(f"⚠️ No se pudo activar TeaCache: {e}")

    # Seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Generar vídeo
    logger.info("\n"+"="*60)
    logger.info(f"🎬 Generando vídeo")
    logger.info(f"Prompt: {args.prompt[:80]}...")
    logger.info(f"Resolución: {args.width}x{args.height}, frames: {args.video_length}")
    logger.info("="*60)

    try:
        start_time = time.time()

        if args.smart_memory:
            # Paso 1: Generar latentes con el Transformer
            logger.info("\n📐 Paso 1/3: Generando latentes (Transformer)...")
            print_gpu_memory()

            with torch.no_grad():
                if args.device == 'cuda':
                    with torch.cuda.amp.autocast(dtype=weight_dtype):
                        output = pipeline(
                            prompt=args.prompt,
                            negative_prompt=args.negative_prompt,
                            height=args.height,
                            width=args.width,
                            num_frames=args.video_length,  # <- Usar num_frames (no cambiar)
                            num_inference_steps=args.num_inference_steps,
                            guidance_scale=args.guidance_scale,
                            generator=torch.Generator(device=args.device).manual_seed(args.seed),
                            output_type="latent",  # Solo latentes
                            return_dict=True,
                        )
                else:
                    output = pipeline(
                        prompt=args.prompt,
                        negative_prompt=args.negative_prompt,
                        height=args.height,
                        width=args.width,
                        num_frames=args.video_length,
                        num_inference_steps=args.num_inference_steps,
                        guidance_scale=args.guidance_scale,
                        generator=torch.Generator(device=args.device).manual_seed(args.seed),
                        output_type="latent",
                        return_dict=True,
                    )

            # Obtener los latentes de output
            latents = output.videos if hasattr(output, 'videos') else output[0]

            # Asegurar tensor
            if isinstance(latents, np.ndarray):
                latents = torch.from_numpy(latents).to(device=args.device, dtype=weight_dtype)
            elif isinstance(latents, torch.Tensor):
                latents = latents.to(device=args.device, dtype=weight_dtype)
            else:
                logger.error(f"Tipo de latentes inesperado: {type(latents)}")
                raise TypeError(f"Se esperaba tensor o numpy array, se obtuvo {type(latents)}")

            inference_time = time.time() - start_time
            logger.info(f"✅ Latentes generados en {inference_time:.2f}s")
            print_gpu_memory()

            # Paso 2: Intercambio de memoria
            logger.info("\n🔄 Paso 2/3: Intercambio de memoria...")
            logger.info("Moviendo Transformer a CPU...")
            pipeline.transformer = pipeline.transformer.to('cpu')
            if hasattr(pipeline, 'text_encoder'):
                pipeline.text_encoder = pipeline.text_encoder.to('cpu')
            torch.cuda.empty_cache()
            logger.info("✅ Memoria GPU liberada")
            print_gpu_memory()

            logger.info("Moviendo VAE a GPU...")
            pipeline.vae = pipeline.vae.to(args.device)

            # Activar VAE tiling si está disponible
            if args.enable_vae_tiling and hasattr(pipeline.vae, 'enable_tiling'):
                pipeline.vae.enable_tiling()
                logger.info("✅ VAE tiling activado")

            print_gpu_memory()

            # Paso 3: Decodificar latentes
            logger.info("\n🎨 Paso 3/3: Decodificando latentes (VAE)...")
            decode_start = time.time()

            with torch.no_grad():
                if args.device == 'cuda':
                    with torch.cuda.amp.autocast(dtype=weight_dtype):
                        sample = pipeline.decode_latents(latents)
                else:
                    sample = pipeline.decode_latents(latents)

            decode_time = time.time() - decode_start
            logger.info(f"✅ Vídeo decodificado en {decode_time:.2f}s")

            # Volver VAE a CPU
            pipeline.vae = pipeline.vae.to('cpu')
            torch.cuda.empty_cache()

        else:
            # Generación tradicional (todo en uno)
            logger.info("\n🎬 Generando vídeo (modo tradicional)...")
            with torch.no_grad():
                if args.device == 'cuda':
                    with torch.cuda.amp.autocast(dtype=weight_dtype):
                        output = pipeline(
                            prompt=args.prompt,
                            negative_prompt=args.negative_prompt,
                            height=args.height,
                            width=args.width,
                            num_frames=args.video_length,  # <- num_frames, no video_length
                            num_inference_steps=args.num_inference_steps,
                            guidance_scale=args.guidance_scale,
                            generator=torch.Generator(device=args.device).manual_seed(args.seed),
                        )
                else:
                    output = pipeline(
                        prompt=args.prompt,
                        negative_prompt=args.negative_prompt,
                        height=args.height,
                        width=args.width,
                        num_frames=args.video_length,
                        num_inference_steps=args.num_inference_steps,
                        guidance_scale=args.guidance_scale,
                        generator=torch.Generator(device=args.device).manual_seed(args.seed),
                    )
                # Extraer de output.videos si existe, si no, primer elemento
                sample = output.videos if hasattr(output, 'videos') else output[0]

        total_time = time.time() - start_time
        logger.info(f"\n✅ ¡Vídeo generado con éxito en {total_time:.2f}s!")
        print_gpu_memory()

        # Guardar vídeo
        logger.info(f"\n💾 Guardando vídeo en {args.output_path}...")

        # Convertir tensor a numpy
        if isinstance(sample, torch.Tensor):
            video_np = sample.cpu().numpy()
        else:
            video_np = sample

        # Debug shape
        logger.info(f"Forma del vídeo antes de procesar: {video_np.shape}")

        # Formato correcto [T, H, W, C]
        if video_np.ndim == 5:
            if video_np.shape[0] == 1:
                video_np = video_np.squeeze(0)
                logger.info(f"Dimensión batch eliminada: {video_np.shape}")

        if video_np.ndim == 4:
            if video_np.shape[-1] not in [1, 3, 4]:
                if video_np.shape[1] in [1, 3, 4]:  # [T, C, H, W]
                    video_np = video_np.transpose(0, 2, 3, 1)
                    logger.info(f"Transpuesta de [T, C, H, W] a [T, H, W, C]: {video_np.shape}")
                elif video_np.shape[0] in [1, 3, 4]:  # [C, T, H, W]
                    video_np = video_np.transpose(1, 2, 3, 0)
                    logger.info(f"Transpuesta de [C, T, H, W] a [T, H, W, C]: {video_np.shape}")

        logger.info(f"Forma final del vídeo: {video_np.shape}")

        # Normalizar a [0, 255]
        if video_np.max() <= 1.0:
            video_np = (video_np * 255).astype(np.uint8)
        else:
            video_np = video_np.astype(np.uint8)

        imageio.mimwrite(args.output_path, video_np, fps=8, codec='h264')

        logger.info("="*60)
        logger.info(f"🎉 ¡ÉXITO! Vídeo guardado en {args.output_path}")
        logger.info(f"⏱️ Tiempo total de generación: {total_time:.2f}s")
        if args.smart_memory and 'inference_time' in locals():
            logger.info(f"   - Inferencia: {inference_time:.2f}s")
            logger.info(f"   - Decodificación: {decode_time:.2f}s")
        logger.info("="*60)

    except Exception as e:
        logger.error(f"\n❌ Error durante la generación: {e}")
        import traceback
        traceback.print_exc()

        # Sugerencias de diagnóstico
        if "weight is None" in str(e):
            logger.error("\n💡 Pista: ¡No se han cargado correctamente los pesos del modelo!")
            logger.error("Comprueba que el archivo GGUF contiene todos los pesos necesarios.")
        elif "GGMLTensor" in str(e):
            logger.error("\n💡 Pista: ¡Problema de compatibilidad GGMLTensor!")
            logger.error("El formato tensor de GGUF no es compatible con esta operación.")
        elif "out of memory" in str(e).lower():
            logger.error("\n💡 Pista: ¡Sin memoria!")
            logger.error("Prueba activando --smart_memory o reduciendo resolución/frames.")
        raise


if __name__ == "__main__":
    main()

