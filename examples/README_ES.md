# CineWan - Wan Transformer GGUF

## Descripción

Este proyecto contiene un ejemplo práctico y funcional de cómo trabajar con modelos Wan 
en formato GGUF, utilizando el pipeline base de 
[VideoX-Fun](https://github.com/aigc-apps/VideoX-Fun/) para la carga e inferencia de 
modelos de video [Wan 2.1-2.2](https://github.com/Wan-Video/Wan2.1).

El objetivo es mostrar cómo adaptar un modelo Wan a pesos cuantizados GGUF y ejecutar 
inferencia eficiente, incluso en GPUs con recursos limitados.

## Rendimiento

Con la inferencia de modelos GGUF he conseguido ejecutar, en una tarjeta RTX 3090, una 
carga ultra rápida en **20 segundos** del Transformer de 14B de Wan 2.1, incluyendo 
todos los componentes (tokenizer, VAE, etc.), y una inferencia de vídeo de las 
siguientes características en **40 segundos**:

- **Resolución**: 512x320 píxeles
- **Frames**: 81 frames
- **Pasos**: 4 steps
- **Modelo**: LightX2V destilado en formato GGUF

Modelo utilizado: 
[Wan2.1_T2V_14B_LightX2V_StepCfgDistill_VACE-GGUF](https://huggingface.co/QuantStack/Wan2.1_T2V_14B_LightX2V_StepCfgDistill_VACE-GGUF)

### Ventajas de GGUF

La principal ventaja de cargar modelos en formato GGUF es la **alta velocidad de carga**, 
aunque se pierda ligeramente algo de definición en comparación con los modelos 
originales sin cuantizar.

## Estructura del Proyecto

### `wan_transformer_gguf.py`

Contiene la arquitectura adaptada del modelo Wan, específicamente reconstruida para ser 
compatible con el sistema de pesos GGUF.

**Características principales:**

- Replica exactamente el `WanTransformer3DModel` de VideoXFun
- Sustituye las capas pesadas estándar (`Linear` y `Conv3d`) por equivalentes 
  optimizadas para GGUF a través de `GGMLOps`
- Permite cargar directamente pesos cuantizados de manera eficiente
- Respeta particularidades arquitectónicas como `AdaLayerNorm` (normalización adaptativa)

### `run_gguf_t2v.py`

Script de ejemplo para ejecutar generación de vídeo texto-a-vídeo utilizando la 
arquitectura adaptada y el cargador GGUF.

**Funcionalidades:**

- Gestión inteligente de memoria para generar vídeos de alta resolución/duración en 
  GPUs de VRAM limitada
- Configuración completa de parámetros (prompt, tamaño, sampler, pasos, etc.)
- Swap automático de módulos entre CPU/GPU de forma transparente

## Funcionamiento del Sistema

### 1. Arquitectura Adaptada (`wan_transformer_gguf.py`)

La clase `WanTransformerGGUF` reproduce la arquitectura del modelo Wan original, 
utilizando capas optimizadas para pesos GGUF (cuantizados).

**Puntos clave:**

- Todas las capas con pesos importantes se crean con `self.ops = GGMLOps()`
- Capas de normalización críticas configuradas como `LayerNorm` sin pesos aprendibles
- Las capas de atención se reconstruyen para emplear capas GGUF
- `create_wan_transformer_gguf` asigna los pesos mapeando cada tensor del state_dict

### 2. Script de Ejecución (`run_gguf_t2v.py`)

Demuestra el uso del modelo adaptado en un flujo completo de texto a vídeo.

**Gestión inteligente de memoria:**

1. Mueve solo el transformer y text encoder a GPU para generar los latentes
2. Hace swap de memoria: libera VRAM pasando esos módulos a CPU
3. Sube el VAE a GPU para decodificar los latentes en el vídeo final
4. Evita que coincidan los módulos más pesados en VRAM simultáneamente

## Uso

### 1. Preparación

Necesitas:
- Un archivo de modelo Wan en formato `.gguf`
- La configuración YAML
- Los componentes base (VAE, text_encoder)

### 2. Ejecución del Script

```bash
python run_gguf_t2v.py \
  --gguf_path path/al/modelo.wan.gguf \
  --base_model_path path/al/base_model/ \
  --config_path path/al/config.yaml \
  --prompt "Un dron sobrevolando una ciudad futurista" \
  --output_path ejemplo_video.mp4 \
  --height 320 --width 512 --video_length 17 --num_inference_steps 30
```

### 3. Integración en Código Propio (Opcional)

```python
from wan_transformer_gguf import create_wan_transformer_gguf
model = create_wan_transformer_gguf(config, state_dict)
```

## Distribución de Responsabilidades

### Script `run_gguf_t2v.py` (Jefe de Obra)
- **Orquestación y configuración**
- Sabe DÓNDE están los archivos
- Decide QUÉ se va a generar (prompt, resolución, etc.)
- Delega la carga llamando a `load_gguf_pipeline`
- Gestiona el pipeline una vez cargado (mueve componentes entre CPU/GPU)

### Módulo `gguf_loader` (Equipo Especializado)
- **Ejecución de la carga**
- Sabe CÓMO cargar cada componente
- Ejecuta las llamadas `.from_pretrained` para VAE y Tokenizer
- Ejecuta la lógica especializada para cargar el Transformer desde GGUF
- Ensambla todas las piezas y devuelve el producto terminado

## ⚠️ Importante sobre el Sampler

**¡ATENCIÓN!** Es fundamental utilizar el sampler **UniPC (Flow_Unipc)** cuando trabajes 
con modelos Wan 2.1 o 2.2.

No uses otro tipo de sampler salvo los recomendados (UniPC o, en algunos casos, Flow), 
porque otros samplers pueden producir:
- Ruido en el vídeo resultante
- Problemas de interpretación de latentes por parte del VAE

Esta advertencia es especialmente importante para obtener la mejor calidad y evitar 
resultados inesperados.

## Notas Importantes

- **Compatibilidad**: Diseñado para VideoXFun (Wan 2.1 y 2.2)
- **Diferencias**: Arquitectura idéntica a nivel lógico, pero con capas específicas 
  para pesos cuantizados
- **Memoria**: Sistema de swap CPU/GPU permite inferencia eficiente en GPUs con poca VRAM

## Referencias

- [Wan VideoXFun](https://github.com/aigc-apps/VideoX-Fun/)
- [Wan 2.1](https://github.com/Wan-Video/Wan2.1)
- Arquitectura GGUF optimizada por [@acaimari](https://github.com/acaimari)
- Basado en el código original [ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF/) 
  creado por City96

## Licencia

Este proyecto está bajo licencia **Apache 2.0**. Todos los componentes utilizados 
(VideoX-Fun, ComfyUI-GGUF, y el código de este proyecto) están bajo licencia Apache 2.0.
