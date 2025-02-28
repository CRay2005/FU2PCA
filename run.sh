export MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"
export INSTANCE_DIR="lora_imgs/fu_imgs"
export OUTPUT_DIR="lora-trained-xl-fu"
export VAE_PATH="madebyollin/sdxl-vae-fp16-fix"
export HF_ENDPOINT="https://hf-mirror.com"

accelerate launch train_dreambooth_lora_sdxl.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --pretrained_vae_model_name_or_path=$VAE_PATH \
  --output_dir=$OUTPUT_DIR \
  --mixed_precision="bf16" \
  --instance_prompt="a photo of <cn-fu-pca>" \
  --resolution=1024 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --learning_rate=1e-4 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=1600 \
  --seed="0" 





~/diffusers/examples/dreambooth# 
ls lora-trained-xl*/*safetensors       
lora-trained-xl-fish/pytorch_lora_weights.safetensors  
lora-trained-xl-flower/pytorch_lora_weights.safetensors  
lora-trained-xl/pytorch_lora_weights.safetensors
