from transformers import AutoModel
from peft import LoraConfig, get_peft_model

model = AutoModel.from_pretrained("google/siglip-base-patch16-224")
config = LoraConfig(
    r=8, target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
    task_type=None  # or remove completely
)
model.vision_model = get_peft_model(model.vision_model, config)
print("LoRA applied successfully!")