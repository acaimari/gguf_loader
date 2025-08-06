# Cargador GGUF para Modelos Wan V1.0

Un cargador de alto rendimiento para modelos de generación de vídeo Wan en formato 
`.gguf`, diseñado para reducir drásticamente el consumo de memoria VRAM y RAM durante 
la inferencia.

## Índice

- [¿Qué es gguf_loader?](#qué-es-gguf_loader)
- [Características Principales](#características-principales)
- [Rendimiento](#rendimiento)
- [Instalación](#instalación)
- [Uso Rápido: Integración en 3 Pasos](#uso-rápido-integración-en-3-pasos)
- [Uso Avanzado: Control Granular](#uso-avanzado-control-granular)
- [Modelos GGUF Compatibles](#modelos-gguf-compatibles)
- [Preguntas Frecuentes](#preguntas-frecuentes)
- [Licencia](#licencia)
- [Agradecimientos](#agradecimientos)

## ¿Qué es gguf_loader?

Los modelos de generación de vídeo como la familia Wan son extremadamente potentes, pero 
su gran tamaño (especialmente el componente Transformer) exige una cantidad muy elevada 
de VRAM, limitando su uso a hardware de gama alta.

Este módulo soluciona ese problema implementando una **estrategia de decuantización 
sobre la marcha**. Permite cargar los pesos del Transformer desde un archivo `.gguf` 
cuantizado (4/5/8 bits), manteniendo el modelo en un formato de bajo consumo en memoria 
y decodificando únicamente las capas necesarias en la VRAM en el momento preciso de su 
ejecución.

El resultado es una **reducción drástica del pico de memoria VRAM** requerida, haciendo 
posible ejecutar estos modelos en GPUs de consumo.

## Características Principales

- ⚡ **Reducción Masiva de Memoria**: Reduce el consumo de VRAM del Transformer en un 
  70% o más, dependiendo del nivel de cuantización
- 🎮 **Inferencia en GPUs de Consumo**: Permite la ejecución de modelos como Wan-14B 
  en tarjetas gráficas con VRAM limitada (ej. 12-16 GB)
- 🔌 **Integración Sencilla**: Diseñado para reemplazar la carga estándar de 
  `.safetensors` con una sola función
- 🧠 **Gestión Inteligente de Memoria**: Incluye scripts que demuestran cómo cargar 
  componentes de forma secuencial para minimizar la huella de memoria
- 🛠️ **Flexible**: Ofrece tanto carga "todo en uno" como funciones granulares para 
  control total

## Rendimiento

Con la inferencia de modelos GGUF he conseguido ejecutar, en una tarjeta **RTX 3090**, 
una carga ultra rápida en **20 segundos** del Transformer de 14B de Wan 2.1, 
incluyendo todos los componentes (tokenizer, VAE, etc.), y una inferencia de vídeo con 
las siguientes características en **40 segundos**:

- **Resolución**: 512x320 píxeles
- **Frames**: 81 frames
- **Pasos**: 4 steps
- **Modelo**: LightX2V destilado en formato GGUF

### Ventajas de GGUF

La principal ventaja de cargar modelos en formato GGUF es la **alta velocidad de carga**, 
aunque se pierda ligeramente algo de definición en comparación con los modelos 
originales sin cuantizar.

## Instalación

### Requisitos Previos

- Python 3.8 o superior
- Un proyecto que actualmente cargue el modelo Wan utilizando `diffusers`
- Un archivo de modelo Wan en formato `.gguf`

### Pasos de Instalación

1. **Clona el repositorio:**
   ```bash
   git clone https://github.com/acaimari/gguf_loader.git
   cd gguf_loader
   ```

2. **Instala las dependencias:**
   ```bash
   pip install torch diffusers transformers imageio omegaconf-py
   pip install gguf
   ```

## Uso Rápido: Integración en 3 Pasos

Esta guía asume que ya tienes un proyecto funcional que carga un modelo Wan con diffusers.

### 1. Copia el Módulo a tu Proyecto

Copia el directorio completo `gguf_loader/` en la raíz de tu proyecto. Asegúrate de 
mantener la estructura de carpetas para evitar errores de importación.

### 2. Localiza tu Código de Carga Actual

Busca en tu código la parte donde ensamblas el pipeline. Probablemente se parezca a esto:

**Código ANTERIOR (cargando con .safetensors):**
```python
from diffusers import WanPipeline, WanTransformer3DModel, AutoencoderKL

# ... Carga de VAE, Tokenizer, Scheduler ...

print("Cargando Transformer desde safetensors...")
transformer = WanTransformer3DModel.from_pretrained(
    "Wan-AI/Wan2.1-T2V-14B-Diffusers",
    subfolder="transformer"
)

pipe = WanPipeline(
    transformer=transformer,
    vae=vae,
    # ... otros componentes
)
```

### 3. Reemplaza la Carga por `load_gguf_pipeline`

**Código NUEVO (cargando con .gguf):**
```python
from gguf_loader import load_gguf_pipeline
from gguf_loader.scripts.run_gguf_t2v import FlowUniPCMultistepScheduler

# Configura tu scheduler
scheduler = FlowUniPCMultistepScheduler(...)

print("Cargando pipeline completo usando GGUF Loader...")
pipe = load_gguf_pipeline(
    gguf_path="ruta/a/tu/modelo.gguf",
    base_model_path="Wan-AI/Wan2.1-T2V-14B-Diffusers",
    config_path="ruta/a/tu/config.yaml",
    scheduler=scheduler
)

print("¡Pipeline listo para usar con GGUF!")
```

¡Eso es todo! El pipeline ahora utilizará el Transformer optimizado para GGUF.

## Uso Avanzado: Control Granular

Si prefieres cargar el VAE y otros componentes por tu cuenta y solo usar el módulo para 
el Transformer, utiliza `load_gguf_transformer`:

```python
from gguf_loader import load_gguf_transformer
from diffusers import WanPipeline, AutoencoderKL

# 1. Carga tus componentes como siempre
vae = AutoencoderKL.from_pretrained(
    "Wan-AI/Wan2.1-T2V-14B-Diffusers", 
    subfolder="vae"
)
# ... Carga de Tokenizer, Text Encoder, Scheduler ...

# 2. Carga SOLO el Transformer usando el módulo GGUF
gguf_transformer = load_gguf_transformer(
    gguf_path="ruta/a/tu/modelo.gguf"
)

# 3. Ensambla el pipeline manualmente
pipe = WanPipeline(
    transformer=gguf_transformer,
    vae=vae,
    # ... el resto de tus componentes
)
```

Para un ejemplo completo con gestión inteligente de memoria (CPU/GPU swap), consulta 
el script `examples/run_gguf_t2v.py`.

## Modelos GGUF Compatibles

Puedes encontrar modelos Wan en formato GGUF en los siguientes repositorios de 
Hugging Face:

- [city96](https://huggingface.co/city96) - Modelos originales y conversiones
- [QuantStack](https://huggingface.co/QuantStack) - Incluye el modelo LightX2V destilado
- [Kijai](https://huggingface.co/Kijai) - Variantes adicionales

**Modelo recomendado:** 
[Wan2.1_T2V_14B_LightX2V_StepCfgDistill_VACE-GGUF](https://huggingface.co/QuantStack/Wan2.1_T2V_14B_LightX2V_StepCfgDistill_VACE-GGUF)

### ⚠️ Importante sobre el Sampler

**¡ATENCIÓN!** Es fundamental utilizar el sampler **UniPC (Flow_Unipc)** cuando 
trabajes con modelos Wan 2.1 o 2.2. No uses otros samplers ya que pueden producir:

- Ruido en el vídeo resultante
- Problemas de interpretación de latentes por parte del VAE

## Preguntas Frecuentes

**¿Qué ventajas tiene el formato .gguf respecto a .safetensors?**
- Mejor eficiencia en la carga (hasta 10x más rápido)
- Menor consumo de memoria (reducción del 70%)
- Soporte para cuantización avanzada
- Menor dependencia de librerías externas

**¿Es compatible con todos los modelos Wan?**
Solo es compatible con modelos entrenados o convertidos al formato `.gguf`. 
Actualmente soporta Wan 2.1 y 2.2.

**¿Dónde puedo consultar ejemplos de integración?**
Consulta la carpeta `examples/` para ejemplos completos de integración con Wan 2.1.

**¿Qué GPU mínima necesito?**
Con GGUF puedes ejecutar modelos Wan-14B en GPUs con al menos 12GB de VRAM 
(como RTX 3060 12GB, RTX 4070, etc.).

## Licencia

Este proyecto se distribuye bajo la **licencia Apache 2.0**. Todos los componentes 
utilizados están bajo la misma licencia.

## Agradecimientos

Este trabajo está inspirado y adaptado del excelente proyecto 
[ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF/) de 
[@city96](https://github.com/city96). Agradecemos enormemente su contribución a la 
comunidad de código abierto.

## Soporte

Si tienes cualquier duda o necesitas soporte, abre una **issue** en este repositorio 
o contacta con [@acaimari](https://github.com/acaimari).
