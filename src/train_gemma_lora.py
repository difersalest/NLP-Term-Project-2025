from unsloth import FastLanguageModel
import torch
from trl import SFTTrainer
from unsloth.chat_templates import train_on_responses_only
from transformers import TrainingArguments
from datasets import Dataset
import pandas as pd
from pathlib import Path
import json
from sklearn.model_selection import train_test_split

# ==============================
# Configuration
# ==============================
class CFG:
    MODEL_NAME = "unsloth/gemma-2-9b-it-bnb-4bit"
    
    # 檔案路徑
    INPUT_FILE = "../data/folds/train_folds.csv"
    OUTPUT_DIR = Path("../data/folds/gemma_lora_final") # 只有一個輸出版
    
    # LoRA 參數
    MAX_SEQ_LENGTH = 1800
    LORA_R = 16
    LORA_ALPHA = 32
    LORA_DROPOUT = 0
    
    # 訓練參數
    EPOCHS = 3         # 資料翻倍後，1個 epoch 絕對夠了
    BATCH_SIZE = 8      # 視顯存調整
    GRAD_ACCUM = 4
    LR = 2e-4
    SEED = 42
    VAL_SIZE = 0.05     # 只留 5% 做驗證，95% 用於訓練

CFG.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

'''
def sandwich_slice(text, max_len, head_ratio=0.25):
    if len(text) <= max_len:
        return text
    
    placeholder = " ... "
    budget = max_len - len(placeholder)
    
    if budget <= 0:
        return text[:max_len]
    
    head_len = int(budget * head_ratio)
    tail_len = budget - head_len
    
    return text[:head_len] + placeholder + text[-tail_len:]
'''

def format_func(row):
    try:
        p = json.loads(row['prompt'])[0]
        a = json.loads(row['response_a'])[0]
        b = json.loads(row['response_b'])[0]
    except:
        p, a, b = str(row['prompt']), str(row['response_a']), str(row['response_b'])

    # Sandwich: 頭尾都保留，切中間
    # Prompt: 25% 頭 + 75% 尾（命令常在尾）
    #p_sand = sandwich_slice(p, max_len=1000, head_ratio=0.25)
    
    # Response: 25% 頭 + 75% 尾（結論在尾）
    #a_sand = sandwich_slice(a, max_len=2000, head_ratio=0.25)
    #b_sand = sandwich_slice(b, max_len=2000, head_ratio=0.25)

    # use original approach since we got worse result on sandwich
    p = p[:512]
    a = a[:1024]
    b = b[:1024]

    instruction = f"""### Prompt
{p_sand}

### Response A
{a_sand}

### Response B
{b_sand}

Which model's answer is better? Directly answer with 'A', 'B', or 'tie'."""

    if row['target'] == 0: response = "A"
    elif row['target'] == 1: response = "B"
    else: response = "tie"

    return f"<start_of_turn>user\n{instruction}<end_of_turn>\n<start_of_turn>model\n{response}<end_of_turn>"

print("Loading Data...")
df = pd.read_csv(CFG.INPUT_FILE)
print(f"Original Data Size: {len(df)}")

# --- 🔥 Data Augmentation (Swap A/B) ---
print("Applying Augmentation (Swapping A/B)...")
df_swap = df.copy()

# 1. 交換內容
df_swap = df_swap.rename(columns={
    'response_a': 'response_b', 
    'response_b': 'response_a',
    # 如果有解析過的 text 欄位也要換，這裡假設我們會重新解析 json
})

# 2. 反轉 Target (0->1, 1->0, 2->2)
target_map = {0: 1, 1: 0, 2: 2}
df_swap['target'] = df_swap['target'].map(target_map)

# 3. 合併
df_aug = pd.concat([df, df_swap], axis=0).reset_index(drop=True)

# 4. 打亂順序 (Shuffle)
df_aug = df_aug.sample(frac=1, random_state=CFG.SEED).reset_index(drop=True)
print(f"Augmented Data Size: {len(df_aug)} (Doubled!)")

# ==============================
# 2. Split Train/Val
# ==============================
# 我們不做 K-Fold，只切一次
train_df, val_df = train_test_split(df_aug, test_size=CFG.VAL_SIZE, random_state=CFG.SEED)
print(f"Training on {len(train_df)} samples, Validating on {len(val_df)} samples")

# Convert to Dataset
train_ds = Dataset.from_pandas(train_df)
val_ds = Dataset.from_pandas(val_df)

# Apply formatting
train_ds = train_ds.map(lambda x: {'text': format_func(x)}, batched=False)
val_ds = val_ds.map(lambda x: {'text': format_func(x)}, batched=False)

# ==============================
# 3. Setup Model & LoRA
# ==============================
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = CFG.MODEL_NAME,
    max_seq_length = CFG.MAX_SEQ_LENGTH,
    dtype = None,
    load_in_4bit = True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r = CFG.LORA_R,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = CFG.LORA_ALPHA,
    lora_dropout = CFG.LORA_DROPOUT,
    bias = "none",
    use_gradient_checkpointing = "unsloth", 
    random_state = CFG.SEED,
)

# ==============================
# 4. Training
# ==============================
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = train_ds,
    eval_dataset = val_ds,
    dataset_text_field = "text",
    max_seq_length = CFG.MAX_SEQ_LENGTH,
    dataset_num_proc = 2,
    packing = False, 
    args = TrainingArguments(
        per_device_train_batch_size = CFG.BATCH_SIZE,
        gradient_accumulation_steps = CFG.GRAD_ACCUM,
        per_device_eval_batch_size = 4,
        warmup_steps = 10,
        max_steps = -1, 
        num_train_epochs = CFG.EPOCHS,
        learning_rate = CFG.LR,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 10,
        # 【修正點】將 evaluation_strategy 改為 eval_strategy
        eval_strategy = "no", 
        
        #eval_steps = 200,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = CFG.SEED,
        output_dir = str(CFG.OUTPUT_DIR),
        report_to = "none",
        save_strategy = "steps",
        save_steps = 100, 
        save_total_limit = 2,
        resume_from_checkpoint=True,
    ),
)
trainer = train_on_responses_only(
    trainer,
    instruction_part = "<start_of_turn>user\n",
    response_part = "<start_of_turn>model\n",
)
print("Starting Training...")

trainer.train(resume_from_checkpoint=True)

# ==============================
# 5. Save Final Adapter
# ==============================
print(f"Saving final adapter to {CFG.OUTPUT_DIR}...")
model.save_pretrained(CFG.OUTPUT_DIR)
tokenizer.save_pretrained(CFG.OUTPUT_DIR)

print("Done! You can now upload this folder to Kaggle.")
