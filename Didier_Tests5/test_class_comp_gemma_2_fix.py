# %% [markdown]
# # Gemma 3: Fine-Tuning for Classification Tasks

# %% [markdown]
# ## Configuration
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
from dotenv import load_dotenv
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers import (
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    DataCollatorWithPadding,
    AutoModelForCausalLM)
import bitsandbytes as bnb
import evaluate
from huggingface_hub import login
import json
from transformers import DataCollatorWithPadding
from transformers import AutoTokenizer, Gemma3ForCausalLM
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training, get_peft_model
from peft import LoraConfig, get_peft_model
import torch.nn.functional as F
from sklearn.metrics import log_loss
from transformers import EarlyStoppingCallback, TrainingArguments, Trainer
from transformers import AutoTokenizer, Gemma3ForCausalLM, BitsAndBytesConfig
from datasets import Dataset, load_dataset, DatasetDict
from peft import (LoraConfig,
                  PeftModel,
                  prepare_model_for_kbit_training,
                  get_peft_model,
                  PeftModelForSequenceClassification,
                  PeftConfig)
from transformers import AutoTokenizer


# %%


# Set the device to physical GPU 3
# Physics server


env_path = "./config/.env"
load_dotenv(dotenv_path=env_path)


# %%
device_map = {"": 0}
gpu_device = 'cuda:0'

# %%

num_gpus = torch.cuda.device_count()
print(f"Found {num_gpus} GPUs available to PyTorch:")
print("-" * 40)

for i in range(num_gpus):
    name = torch.cuda.get_device_name(i)
    mem = torch.cuda.get_device_properties(i).total_memory / 1024**3
    print(f"Device Index {i}: {name} ({mem:.2f} GB)")

print("-" * 40)

# %% [markdown]
# ## Load Libraries

# %%


# %% [markdown]
# ## Logging into hugging face

# %%

# Configure the NTHU proxy directly in Python using the IP address
proxy_url = "http://140.114.63.4:3128"

os.environ['http_proxy'] = proxy_url
os.environ['https_proxy'] = proxy_url
os.environ['HTTP_PROXY'] = proxy_url
os.environ['HTTPS_PROXY'] = proxy_url

print("Proxy configured via IP address.")

# %%
hf_token = os.getenv('HF_TOKEN')
login(hf_token)

# %% [markdown]
# ## Loading and preparing the dataset
#
# As described in the introduction, we’ll use the thesofakillers/jigsaw-toxic-comment-classification-challenge dataset from the Hugging Face dataset library for this demonstration. We begin by loading only the 'train' portion of this dataset. Since we need distinct sets for training, validation, and testing, we'll perform a couple of splits using the datasets library.
#
# The following code executes these steps:

# %%
def augment_data(df):
    # Create a copy for the swapped data
    df_swapped = df.copy()
    
    # Swap the text content
    df_swapped['response_a'] = df['response_b']
    df_swapped['response_b'] = df['response_a']
    
    # Swap the labels
    # 0: A wins, 1: B wins, 2: Tie
    # If A won (0), now B wins (1). If B won (1), now A wins (0). Tie (2) stays Tie (2).
    
    # We need to swap the one-hot columns if you use them, or the integer label
    # Assuming your dataframe has the original columns from train.csv:
    df_swapped['winner_model_a'] = df['winner_model_b']
    df_swapped['winner_model_b'] = df['winner_model_a']
    # 'winner_tie' remains the same
    
    # Concatenate original and swapped
    df_augmented = pd.concat([df, df_swapped], ignore_index=True)
    
    # Shuffle the dataset so batches contain a mix
    df_augmented = df_augmented.sample(frac=1, random_state=42).reset_index(drop=True)
    
    return df_augmented

# Augment training data by swapping responses order
df_train_pandas = pd.read_csv("kaggle/input/lmsys-chatbot-arena/train.csv")
df_aug = augment_data(df_train_pandas)

dataset_preference = Dataset.from_pandas(df_aug)

# dataset_preference = Dataset.from_pandas(
#     pd.read_csv("kaggle/input/lmsys-chatbot-arena/train.csv"))
dataset_preference_test = Dataset.from_pandas(
    pd.read_csv("kaggle/input/lmsys-chatbot-arena/test.csv"))

dataset_preference = dataset_preference.train_test_split(
    test_size=0.025, seed=42,)

dataset_preference = DatasetDict({
    'train': dataset_preference['train'],
    'valid': dataset_preference['test'],
    # 'test': dataset_preference_test
})

dataset_preference


# %%
# dataset_toxic = load_dataset("thesofakillers/jigsaw-toxic-comment-classification-challenge")
# dataset_toxic = dataset_toxic['train']
# dataset_toxic = dataset_toxic.train_test_split(test_size=0.25,seed=42,)

# test_valid  = dataset_toxic['test'].train_test_split(test_size=0.5)

# dataset_toxic = DatasetDict({
#     'train': dataset_toxic['train'].select(range(1000)),
#     'valid': test_valid['train'].select(range(100)),
#     'test': test_valid['test'].select(range(100))})

# dataset_toxic

# %% [markdown]
# To quickly examine the data’s structure, especially the comments and their labels, we can convert a portion of it, like the training set, into a Pandas DataFrame:

# %%
df = pd.DataFrame(dataset_preference['train'])
df.head()

# %% [markdown]
# This inspection shows the comment_text column alongside the various toxicity labels such as toxic, severe_toxic, obscene etc. These labels are already one-hot encoded. For instance, some comments might be flagged across multiple toxic categories, while many others will display zeros for all labels, indicating they are not hits any category.

# %% [markdown]
# ## Tokenization: Preparing Data for Gemma 3
#
# The next stage in our pipeline is tokenization. This process converts the raw text comments from our dataset into a numerical format that the Gemma 3 model can understand and process. For this tutorial, we’re working with the google/gemma-3-4b-it model. The first step is to load its corresponding tokenizer from the Hugging Face Hub. When loading the tokenizer, we'll specify padding_side='right' and add_bos=True to include a beginning-of-sequence token, often beneficial for Gemma models (check the report Table 4).
#
# A important part of preparing for a multilabel classification task is creating a clear mapping from our category names to numerical indices. This is achieved with a simple Python dictionary, class2id.

# %%
hugging_face_model_id = "google/gemma-3-1b-it"  # google/gemma-3-4b-it

tokenizer = AutoTokenizer.from_pretrained(hugging_face_model_id,
                                          padding_side='right',
                                          device_map=device_map,
                                          add_bos=True)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# class2id = {'winner_model_a':0,'winner_model_b':1,'winner_tie':2}
# id2class = {v: k for k, v in class2id.items()}
class2id = {'winner_model_a': 0, 'winner_model_b': 1, 'winner_tie': 2}
id2class = {0: 'winner_model_a', 1: 'winner_model_b', 2: 'winner_tie'}

# %% [markdown]
# With the tokenizer ready, we define a preprocess_function. This function will take each sample from our dataset, tokenize its comment_text, and crucially, reformat its multiple binary labels into a single list of 0 or 1 corresponding to our class2id mapping. We'll apply this function across our entire dataset using the .map() method.
#
# After tokenization and selecting only the essential columns (input_ids, attention_mask, and our newly created labels), the DatasetDict will look something like this, containing the processed data ready for the model:

# %%
# def preprocess_function(sample):
#     labels = []
#     for class_ in class2id.keys():
#         labels.append(sample[class_])

#     sample = tokenizer(f"""# **Based on the following prompt choose which of the two responses you think humans would prefer the most:** \n
#     ## **Prompt:**
#     `{sample['prompt']}`\n
#     ## **Response A:**
#     `{sample['response_a']}`\n
#     ## **Response B:**
#     `{sample['response_b']}`""",
#                        truncation=True)
#     sample['labels'] = labels
#     return sample


# dataset_preference_tokenized = dataset_preference.map(preprocess_function)
# dataset_preference_tokenized = dataset_preference_tokenized.select_columns(['input_ids','attention_mask','labels'])
# dataset_preference_tokenized

# %%

# --- Configuration ---
# Set the maximum context length (e.g., 8192 for Gemma 3 or specific hardware limits)
MAX_LENGTH = 8192
# Buffer for the template text (headers like "## Prompt:", special tokens, etc.)
TEMPLATE_BUFFER = 100
# Calculate available tokens for actual content
AVAILABLE_TOKENS = MAX_LENGTH - TEMPLATE_BUFFER

# Define ratios based on the "Sandwich" strategy request
# 20% for Prompt (Context/Intent), 40% for Resp A, 40% for Resp B (Conclusions)
PROMPT_RATIO = 0.2
RESP_RATIO = 0.4


# def preprocess_function(sample):
#     # labels = []
#     # for class_ in class2id.keys():
#     #     labels.append(sample[class_])
#     # We look for which column is set to 1
#     if sample['winner_model_a'] == 1:
#         label = 0
#     elif sample['winner_model_b'] == 1:
#         label = 1
#     else:
#         label = 2  # winner_tie

#     # Parsing and Concatenation
#     try:
#         prompt_text = "\n".join(json.loads(sample['prompt']))
#         resp_a_text = "\n".join(json.loads(sample['response_a']))
#         resp_b_text = "\n".join(json.loads(sample['response_b']))
#     except (json.JSONDecodeError, TypeError):
#         # Fallback if data is not a valid JSON string
#         prompt_text = str(sample['prompt'])
#         resp_a_text = str(sample['response_a'])
#         resp_b_text = str(sample['response_b'])

#     # Tokenization for Length Calculation
#     # We tokenize raw parts to check their lengths against our budget
#     # add_special_tokens=False because we just want to count content IDs
#     prompt_ids = tokenizer(prompt_text, add_special_tokens=False)['input_ids']
#     resp_a_ids = tokenizer(resp_a_text, add_special_tokens=False)['input_ids']
#     resp_b_ids = tokenizer(resp_b_text, add_special_tokens=False)['input_ids']

#     # Sandwich / Budget Allocation

#     # Calculate max tokens allowed per section
#     max_prompt_len = int(AVAILABLE_TOKENS * PROMPT_RATIO)
#     max_resp_len = int(AVAILABLE_TOKENS * RESP_RATIO)

#     # Truncate Prompt: Keep the START (Head) -> Preserves Intent
#     if len(prompt_ids) > max_prompt_len:
#         prompt_ids = prompt_ids[:max_prompt_len]

#     # Truncate Responses: Keep the END (Tail) -> Preserves Conclusion/Success State
#     if len(resp_a_ids) > max_resp_len:
#         resp_a_ids = resp_a_ids[-max_resp_len:]  # Slice from end

#     if len(resp_b_ids) > max_resp_len:
#         resp_b_ids = resp_b_ids[-max_resp_len:]  # Slice from end

#     # Decode back to Text

#     final_prompt = tokenizer.decode(prompt_ids, skip_special_tokens=True)
#     final_resp_a = tokenizer.decode(resp_a_ids, skip_special_tokens=True)
#     final_resp_b = tokenizer.decode(resp_b_ids, skip_special_tokens=True)

#     # Final Formatting and Tokenization
#     # Prompt changed to follow what 13th place solution did
#     formatted_input = f"""# **Look at the following prompt and the two model responses:** \\n 
#     ## **Prompt:**
#     `{final_prompt}`\\n
#     ## **Response Model A:**
#     `{final_resp_a}`\\n
#     ## **Response Model B:**
#     `{final_resp_b}`
#     # **Your task is to pick the best response between Model A and Model B or declare a Tie. Answer only with 'A', 'B', or 'Tie'. Think carefully before answering.**"""

#     # Final tokenization with the hard cap to ensure we never exceed physical limits
#     sample_tokenized = tokenizer(formatted_input,
#                                  truncation=True,
#                                  max_length=MAX_LENGTH)

#     # sample_tokenized['labels'] = labels
#     sample_tokenized['labels'] = label
#     return sample_tokenized

def preprocess_function(sample):
    # Label Extraction
    # Convert one-hot encoded columns to a single integer label
    if sample['winner_model_a'] == 1:
        label = 0
    elif sample['winner_model_b'] == 1:
        label = 1
    else:
        label = 2  # winner_tie

    # Parse Data
    try:
        prompts = json.loads(sample['prompt'])
        resps_a = json.loads(sample['response_a'])
        resps_b = json.loads(sample['response_b'])
    except (json.JSONDecodeError, TypeError):
        # Fallback if data is not a valid JSON string
        prompts = [str(sample['prompt'])]
        resps_a = [str(sample['response_a'])]
        resps_b = [str(sample['response_b'])]

    # Setup Constants & Overhead
    # The instruction prompt at the end
    instruction_text = (
        "\n# **Your task is to pick the best response between Model A and Model B or declare a Tie. "
        "Answer only with 'A', 'B', or 'Tie'. Think carefully before answering.**"
    )
    # Tokenize instruction to know its cost
    instruction_ids = tokenizer(instruction_text, add_special_tokens=False)['input_ids']
    
    # We leave a small buffer for safety and special tokens (BOS/EOS)
    current_budget = MAX_LENGTH - len(instruction_ids) - 10 
    
    formatted_rounds_ids = []
    
    # Zip the rounds together
    turns = list(zip(prompts, resps_a, resps_b))
    total_rounds = len(turns)

    # Interleave & Truncate (Backwards Loop)
    for i, (p_text, a_text, b_text) in enumerate(reversed(turns)):
        round_idx = total_rounds - i
        
        # Construct header
        header_text = f"\n\n## Round {round_idx}:\n"
        
        # Tokenize components (raw content count)
        header_ids = tokenizer(header_text, add_special_tokens=False)['input_ids']
        prompt_ids = tokenizer(f"### Prompt:\n{p_text}", add_special_tokens=False)['input_ids']
        resp_a_ids = tokenizer(f"\n### Response Model A:\n{a_text}", add_special_tokens=False)['input_ids']
        resp_b_ids = tokenizer(f"\n### Response Model B:\n{b_text}", add_special_tokens=False)['input_ids']
        
        # Calculate total size of this round
        round_total_len = len(header_ids) + len(prompt_ids) + len(resp_a_ids) + len(resp_b_ids)
        
        # Check Budget
        if round_total_len <= current_budget:
            # Fits perfectly: Add the full round
            full_round = header_ids + prompt_ids + resp_a_ids + resp_b_ids
            formatted_rounds_ids.insert(0, full_round) # Insert at front
            current_budget -= round_total_len
        
        else:
            # Does NOT fit: Apply Sandwich Logic to this specific round
            available_for_content = current_budget - len(header_ids)
            
            # If space is too small, stop (Left Truncation)
            if available_for_content < 50:
                break
                
            # Distribute remaining budget
            p_limit = int(available_for_content * PROMPT_RATIO)
            a_limit = int(available_for_content * RESP_RATIO)
            b_limit = int(available_for_content * RESP_RATIO)
            
            # Truncate: Prompt Head, Response Tails
            p_ids_cut = prompt_ids[:p_limit]
            a_ids_cut = resp_a_ids[-a_limit:] if len(resp_a_ids) > a_limit else resp_a_ids
            b_ids_cut = resp_b_ids[-b_limit:] if len(resp_b_ids) > b_limit else resp_b_ids
            
            # Build and insert
            sandwiched_round = header_ids + p_ids_cut + a_ids_cut + b_ids_cut
            formatted_rounds_ids.insert(0, sandwiched_round)
            
            # Budget exhausted, stop processing earlier rounds
            break

    # Final Assembly
    # Flatten list of lists
    full_input_ids = [token for round_ids in formatted_rounds_ids for token in round_ids]
    
    # Append Instruction
    full_input_ids += instruction_ids
    
    # Decode back to text to allow the tokenizer to handle final BOS/EOS/Padding cleanly
    final_text_content = tokenizer.decode(full_input_ids, skip_special_tokens=True)
    
    # Add the structural prefix
    final_text = "# **Look at the following conversation history:** " + final_text_content

    # Final Tokenization for Trainer
    # This generates input_ids and attention_mask with the correct padding/truncation
    sample_tokenized = tokenizer(final_text,
                                 truncation=True,
                                 max_length=MAX_LENGTH)

    # Attach the label
    sample_tokenized['labels'] = label
    
    return sample_tokenized

# Apply the revised function
dataset_preference_tokenized = dataset_preference.map(preprocess_function)
dataset_preference_tokenized = dataset_preference_tokenized.select_columns(
    ['input_ids', 'attention_mask', 'labels'])
dataset_preference_tokenized

# %% [markdown]
# To verify our tokenization and label preparation, let’s inspect a single sample from the training set. We’ll look at its raw input_ids and labels, and then decode them back into human-readable tokens and label names:

# %%
sample_index = 3  # Choose any sample index

sample_input_ids = dataset_preference_tokenized['train']['input_ids'][sample_index]
sample_labels = dataset_preference_tokenized['train']['labels'][sample_index]

print('Input data for model:')
print(f"IDs   : {sample_input_ids}")
print(f"Labels: {sample_labels}\n")

print('Input data decoded:')
print(f"Tokens: {tokenizer.decode(sample_input_ids)}")
# Reconstruct the label dictionary for this sample
# decoded_labels = {id2class[i]: sample_labels[i] for i in range(len(sample_labels))}
decoded_labels = {id2class[sample_labels]: sample_labels}
print(f"Label dictionary: {decoded_labels}")

# %% [markdown]
# This confirms that our text has been converted into a sequence of token IDs, and the corresponding labels are correctly formatted as a list. The decoded output further clarifies how the original sentence and its toxicity classifications are represented, ready for model training.

# %% [markdown]
# ## Dynamic Padding with DataCollator
#
# The comments text which we’re using, the length of individual text samples will inevitably vary. While models can process single samples of differing lengths for inference, training is almost always performed in batches to leverage computational efficiency. However, to combine multiple sequences into a single batch for the model, all sequences within that batch must have a uniform length.
#
# For our purposes, we’ll use DataCollatorWithPadding from the Hugging Face Transformers library. This utility dynamically pads the shorter sequences in each batch with a special <pad> token until they match the length of the longest sequence in that specific batch.
#
# Let’s load the DataCollatorWithPadding and initialize it with our tokenizer:

# %%
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# %% [markdown]
# to observe the data collator in action, we can take a small selection of samples from our tokenized training set — for instance, the first three — and examine their lengths before and after applying the collator.

# %%
sample_batch_ids = dataset_preference_tokenized['train']['input_ids'][0:3]
sample_batch_ids_collator = data_collator(
    dataset_preference_tokenized['train'][:3])['input_ids'][0:3]
print([len(x) for x in sample_batch_ids])
print([len(x) for x in sample_batch_ids_collator])

# length of each sample without datacollator : [74, 37, 159]
# length of each sample with datacollator    :[159, 159, 159]

# %% [markdown]
# As the output demonstrates, our initial batch of three samples had varying lengths. After processing with DataCollatorWithPadding, all samples in the input_ids list now share the same length.

# %% [markdown]
# ## Loading and Adapting the Causal Language Model for Classification
#
# With our data tokenized and ready, we now turn to loading the pre-trained model. As discussed earlier, since we’re working under the premise that a direct Gemma3ForSequenceClassification (or similar) class isn't readily available for our hypothetical "google/gemma-3-4b-it" model, our strategy is to load it as a causal language model—using a class we'll refer to as Gemma3ForCausalLM—and then tailor it for our multi-label classification task. This adaptation primarily involves replacing its original language modeling head with a new classification head specifically designed for the number of toxic comment categories we aim to predict. The model is then loaded using the from_pretrained method. .
#
# Here’s the code to configure quantization and load the base model:

# %%

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16)


model = Gemma3ForCausalLM.from_pretrained(hugging_face_model_id,
                                          torch_dtype=torch.bfloat16,
                                          device_map=gpu_device,
                                          attn_implementation='flash_attention_2',  # eager
                                          quantization_config=bnb_config)

model.lm_head = torch.nn.Linear(model.config.hidden_size, len(
    class2id.keys()), bias=False, device=gpu_device)

# %% [markdown]
# ## Efficient Adaptation with LoRA (Low-Rank Adaptation)
#
# Instead of undertaking the computationally intensive process of fine-tuning all parameters of our large Gemma 3 model, we will employ a more efficient and now classical technique: Low-Rank Adaptation, or LoRA. This Parameter-Efficient Fine-Tuning (PEFT) method keeps the vast majority of the pre-trained model weights frozen. The key idea is to inject small, trainable “adapter” layers into specific existing layers of the model. These adapters learn to modify the model’s activations to suit our specific classification task during training, while the original knowledge of the base model remains largely intact. This significantly reduces the number of parameters that need to be updated, leading to faster training and lower memory requirements.
#
# Before integrating LoRA, we’ll enable gradient checkpointing on our model, a technique that further reduces memory usage during the backward pass,

# %%
model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

# %% [markdown]
# LoRA adapters are typically inserted into the linear layers of a Transformer model. While a helper function like the one below can be used to identify all potential linear layers for LoRA injection (specifically bnb.nn.Linear4bit layers in our 4-bit quantized model, excluding the lm_head), it's also a common and effective practice to use a predefined list of target modules known to yield good performance for a given architecture.

# %%


def find_all_linear_names(model):
    # if args.bits == 4 else (bnb.nn.Linear8bitLt if args.bits == 8 else torch.nn.Linear)
    cls = bnb.nn.Linear4bit
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
        if 'lm_head' in lora_module_names:  # needed for 16-bit
            lora_module_names.remove('lm_head')
    return list(lora_module_names)


modules = find_all_linear_names(model)
modules = ['gate_proj', 'down_proj', 'v_proj',
           'k_proj', 'q_proj', 'o_proj', 'up_proj']

# %% [markdown]
# With our target modules identified (or specified), we define the LoraConfig. This configuration specifies parameters like the rank (r) of the adapter matrices, the LoRA alpha (lora_alpha) scaling factor, the target_modules we just defined, a dropout rate (lora_dropout) for regularization, and importantly, the task_type which is set to "SEQ_CLS" to align with our sequence classification objective.

# %%

lora_config = LoraConfig(
    r=64,
    lora_alpha=16, # 32 before, 16 following solution 4
    target_modules=modules,
    lora_dropout=0.05, # 0.1 before, 0.05 following solution 4
    bias="none",
    task_type="SEQ_CLS")

model = get_peft_model(model, lora_config)

# %% [markdown]
# After applying this configuration using get_peft_model, the model object is now enhanced with LoRA adapters. A call to model.print_trainable_parameters() will clearly demonstrate the efficiency of this approach, showing that only a small fraction of the total parameters are now trainable, drastically reducing the fine-tuning burden.

# %%
model.print_trainable_parameters()
# trainable params: 119,209,984 || all params: 3,999,488,512 || trainable%: 2.9806

# %% [markdown]
# ## Customizing PEFT for Sequence Classification with a Causal LM
#
# This section introduces a key customization that enables us to effectively train our LoRA-adapted causal Gemma 3 model for sequence classification. While PEFT provides PeftModelForSequenceClassification, we'll create a slightly tailored version. This custom class, which we'll call Gemma3ForSequenceClassification, ensures that the forward pass and loss calculation are correctly handled for our specific setup, which uses a base causal model modified for multi-label classification.
#
# The core of this adaptation lies in how we process the model’s outputs. A causal language model, even with a replaced head, produces logits for every token in the sequence. For sequence classification, we are typically interested in a single representation for the entire sequence. A common strategy, adopted here, is to use the logits corresponding to the last token of the sequence as the input to our classification loss function. Our custom class implements this logic and incorporates the appropriate loss function for multi-label tasks, BCEWithLogitsLoss.
#
# Here’s the definition of our Gemma3ForSequenceClassification

# %%


class Gemma3ForSequenceClassification(PeftModelForSequenceClassification):
    def __init__(self, peft_config: PeftConfig, model: AutoModelForCausalLM, adapter_name="default"):
        super().__init__(model, peft_config, adapter_name)
        self.num_labels = model.config.num_labels
        self.problem_type = "single_label_classification"  # Assuming single-label

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            **kwargs):

        return_dict = return_dict if return_dict is not None else self.config.return_dict

        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs)

        # Extract logits from the outputs
        logits = outputs.logits

        # select last "real" token and ignore padding tokens

        sequence_lengths = torch.sum(attention_mask, dim=1)
        last_token_indices = sequence_lengths - 1
        batch_size = logits.shape[0]

        # Get the logits for the last token in the sequence
        logits = logits[torch.arange(
            batch_size, device=logits.device), last_token_indices, :]
        # logits = logits[:, -1, :] # if batch_size = 1

        loss = None
        if labels is not None:
            if self.problem_type == "regression":
                loss_fct = torch.nn.MSELoss()
                loss = loss_fct(logits.squeeze(), labels.squeeze())
            elif self.problem_type == "single_label_classification":
                loss_fct = torch.nn.CrossEntropyLoss()
                loss = loss_fct(
                    logits.view(-1, self.num_labels), labels.view(-1))
            elif self.problem_type == "multi_label_classification":
                loss_fct = torch.nn.BCEWithLogitsLoss()
                loss = loss_fct(logits, labels.float())

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions)

# %% [markdown]
# With this custom class defined, we then instantiate it. We create a PeftConfig object, populating it with the parameters from our lora_config (defined in the previous LoRA setup step). This ensures that our Gemma3ForSequenceClassification wrapper is aware of the LoRA configuration. The model argument passed here is the LoRA-enhanced model we obtained from get_peft_model


# %%
peft_config = PeftConfig(
    peft_type="LORA", task_type="SEQ_CLS", inference_mode=False)
for key, value in lora_config.__dict__.items():
    setattr(peft_config, key, value)

wrapped_model = Gemma3ForSequenceClassification(peft_config, model)
wrapped_model.num_labels = len(class2id.keys())

# %% [markdown]
# ## Implementing a Custom Loss Function (Optional)
#
# While the Hugging Face Trainer and PyTorch provide robust, standard loss functions—and for multi-label classification, torch.nn.BCEWithLogitsLoss is generally the recommended, numerically stable choice—there might be scenarios where you wish to define or understand the loss calculation more explicitly, or perhaps introduce unique modifications. This section demonstrates how to implement a binary cross-entropy loss manually and then integrate it by creating a custom Trainer class. It's important to note that directly implementing sigmoid followed by log calculations can be less numerically stable than combined functions like BCEWithLogitsLoss, but this approach offers a clear view of the underlying mechanics.
#
# First, let’s define our custom binary cross-entropy function. This function will take raw logits and true labels as input. It applies a sigmoid function to the logits to obtain probabilities, clamps these probabilities to a small range (epsilon to 1-epsilon) to avoid log(0) issues for numerical stability, and then calculates the binary cross-entropy loss,
#
# first all define our the function for the binary crossentropy

# %%


def custom_binary_crossentropy_loss(logits, labels, epsilon=1e-7):

    probs = torch.sigmoid(logits)
    probs = torch.clamp(probs, min=epsilon, max=1-epsilon)  # capping values
    loss = -(labels * torch.log(probs) + (1 - labels) * torch.log(1 - probs))
    return torch.mean(loss)

# %% [markdown]
# To utilize this custom_binary_crossentropy_loss function within the standard Hugging Face training workflow, we can create a new class, CustomTrainer, that inherits from the base Trainer class. We then override its compute_loss method to incorporate our custom calculation.

# %%


class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, num_items_in_batch=4, return_outputs=False):
        labels = inputs.get("labels")
        inputs = inputs.to(gpu_device)
        outputs = model(**inputs)
        logits = outputs.logits

        loss = custom_binary_crossentropy_loss(logits, labels)

        return (loss, outputs) if return_outputs else loss

# %% [markdown]
# In this CustomTrainer, the compute_loss method is overridden. It first prepares the inputs and retrieves the true labels. The model then performs a forward pass with the remaining inputs to produce outputs, from which we extract the logits. Our custom_binary_crossentropy_loss function is then called with these logits and the labels to calculate the loss. The method returns the loss, and optionally the model's outputs, aligning with the expected behavior of the Trainer's compute_loss method.
#
# This customized Trainer can now be used in place of the standard Trainer if you wish to proceed with this explicit loss computation method.

# %% [markdown]
# ## Defining Evaluation Metrics
#
# To measure our multi-label classifier’s performance, we’ll use standard metrics: accuracy, F1-score, precision, and recall. The Hugging Face evaluate library conveniently groups these for us. Since our model outputs raw logits, we'll first need a sigmoid function to convert these into probabilities (0 to 1).
#
# The compute_metrics function, designed for the Hugging Face Trainer, handles the main logic. It takes the model's predicted logits and true labels. Inside, it applies the sigmoid function to the logits, then converts these probabilities into binary (0 or 1) predictions using a 0.5 threshold. For multi-label evaluation with these metrics, both the binarized predictions and the true labels are flattened. This approach treats each label for each sample as an independent prediction, allowing us to calculate an overall performance score. These processed arrays are then fed into our combined metric evaluator.


# %%

clf_metrics = evaluate.combine(["accuracy", "f1", "precision", "recall"])

accuracy_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")
precision_metric = evaluate.load("precision")
recall_metric = evaluate.load("recall")


def compute_metrics(eval_pred):
    logits, labels = eval_pred

    # Get Hard Predictions
    predictions = np.argmax(logits, axis=-1)

    # Get Probabilities (For Log Loss)
    # Logits are raw scores
    # Log Loss needs probabilities summing to 1
    # We convert numpy->tensor, apply softmax, then back to numpy
    probs = F.softmax(torch.tensor(logits), dim=-1).numpy()

    # Compute Standard Metrics
    acc = accuracy_metric.compute(predictions=predictions, references=labels)
    f1 = f1_metric.compute(predictions=predictions,
                           references=labels, average="macro")
    prec = precision_metric.compute(
        predictions=predictions, references=labels, average="macro")
    rec = recall_metric.compute(
        predictions=predictions, references=labels, average="macro")

    # Compute Log Loss
    ll_score = log_loss(labels, probs)

    # Combine Results
    results = {**acc, **f1, **prec, **rec}
    results["log_loss"] = ll_score  # Add manually to dictionary

    return results


# %% [markdown]
# ## Start Training
#
# With our model, data, and evaluation components prepared, we’re ready to begin the training process. First, we’ll set up an EarlyStoppingCallback to monitor performance and halt training if the model stops improving, preventing overfitting. We also specify a directory to save our model checkpoints. Additionally, it's often useful to manage tokenizer parallelism by setting an environment variable, which can prevent potential issues with some multi-processing setups.

# %%

# Define early stopping and checkpoint directory
early_stop = EarlyStoppingCallback(early_stopping_patience=3,  # Increased patience slightly
                                   early_stopping_threshold=0.001)  # A small threshold
checkpoints_dir = 'preference_class_gemma_model_1b_fix_2'  # More descriptive name

# Set tokenizer parallelism environment variable
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# %% [markdown]
# Next, we configure TrainingArguments. This object holds a wide array of hyperparameters and settings that control the training loop. Key settings include the output directory for checkpoints, learning rate, batch sizes for training and evaluation, the number of epochs, weight decay for regularization, evaluation and saving strategies, and the metric for identifying the best model (here, eval_loss). We are also enabling mixed-precision training (fp16=True) for efficiency and setting up gradient accumulation.

# %%
training_args = TrainingArguments(
    gradient_checkpointing=False,  # Gradient Checkpointing ist nicht aktiviert
    gradient_checkpointing_kwargs={"use_reentrant": False},
    logging_strategy="steps",
    logging_steps=8,
    # label_names=classes,
    dataloader_num_workers=4,
    output_dir=checkpoints_dir,  # Output directory for checkpoints
    learning_rate=5e-5,  # Learning rate for the optimizer
    per_device_train_batch_size=4,  # Batch size per device, 8 in the beginning
    # Batch size per device for evaluation, 8 in the beginning
    per_device_eval_batch_size=4,
    num_train_epochs=1,  # Number of training epochs
    weight_decay=0.01,  # Weight decay for regularization
    eval_strategy='epoch',  # Evaluate after each epoch
    # eval_steps=8,
    save_strategy="epoch",  # Save model checkpoints after each epoch
    load_best_model_at_end=True,  # Load the best model based on the chosen metric
    push_to_hub=False,  # Disable pushing the model to the Hugging Face Hub
    report_to="tensorboard",  # Disable logging to Weight&Bias
    logging_dir=f"tensorboard_my_model",
    gradient_accumulation_steps=8,  # 4 before
    fp16=False,
    bf16=True,
    warmup_ratio=0.05,
    metric_for_best_model='eval_loss',)  # Metric for selecting the best model

# %% [markdown]
# With the arguments defined, we instantiate the Trainer. This brings together our LoRA-adapted wrapped_model, the training_args, the tokenized training and validation datasets, our data_collator for creating batches, the compute_metrics function, and the early_stop callback.

# %%
trainer = Trainer(
    model=wrapped_model,  # The LoRA-adapted model
    args=training_args,  # Training arguments
    train_dataset=dataset_preference_tokenized['train'],  # Training dataset
    eval_dataset=dataset_preference_tokenized['valid'],  # Evaluation dataset
    # tokenizer=tokenizer,  # Tokenizer for processing text
    data_collator=data_collator,  # Data collator for preparing batches
    compute_metrics=compute_metrics,  # Function to calculate evaluation metrics
    callbacks=[early_stop]  # Optional early stopping callback
)

# %% [markdown]
# Finally, calling trainer.train() launches the fine-tuning process. The trainer will handle the training loop, evaluation, and checkpointing according to our defined arguments.
# This will begin training your custom Gemma 3 classification model. Monitor the logs and evaluation metrics to observe its learning progress.
#

# %%
trainer.train(resume_from_checkpoint=False)

# %% [markdown]
# For the purpose of this demonstration, the dataset was downsampled to a smaller subset of samples. The training process on this reduced dataset successfully showed a clear reduction in the second epoch, alongside corresponding improvements in evaluation metrics such as accuracy and F1-score, indicating that the model was learning effectively from the data.

# %% [markdown]
# ## Making Predictions with the Fine-Tuned Model
#
# Now that our model has been trained, the next step is to use it to make predictions on new, unseen data, such as the comments in our test set. To facilitate this, we’ll first define a helper function, prediction. This function will take a raw text string as input, process it through our trained model, and then return the predicted probabilities for each toxicity category, with the labels sorted by the model's confidence. We'll need our tokenizer, the trained wrapped_model, the id2class mapping (from label indices back to names), and the target device (e.g., "cuda:0") to be available from our previous setup.
#
# The prediction function tokenizes the input text and sends it to the specified device. It then performs inference using our wrapped_model within a torch.no_grad() context to disable gradient calculations, which are unnecessary for prediction and save memory. After obtaining the logits, it applies a sigmoid function to convert them into probabilities. These probabilities are then paired with their corresponding class labels (via id2class), and both are sorted in descending order of probability to clearly show the model's most confident predictions first.

# %%
# def prediction(input_text):
#     inputs          = tokenizer(input_text, return_tensors="pt",).to("cuda:0")
#     with torch.no_grad():
#         outputs = wrapped_model(**inputs).logits
#     y_prob          = np.round(np.array(torch.sigmoid(outputs).tolist()[0]),5)
#     y_sorted_labels = [id2class.get(y) for y  in np.argsort(y_prob)[::-1]]
#     y_prob_sorted   = np.sort(y_prob)[::-1]

#     return y_sorted_labels,y_prob_sorted

# %%
# Configuration 
# MAX_LENGTH = 8192
# TEMPLATE_BUFFER = 200
# AVAILABLE_TOKENS = MAX_LENGTH - TEMPLATE_BUFFER
# PROMPT_RATIO = 0.2
# RESP_RATIO = 0.4

# Helper to Format Input (Sandwich Strategy)

def prepare_inference_input(row):
    # Parse Data
    try:
        prompts = json.loads(row['prompt'])
        resps_a = json.loads(row['response_a'])
        resps_b = json.loads(row['response_b'])
    except (json.JSONDecodeError, TypeError):
        prompts = [str(row['prompt'])]
        resps_a = [str(row['response_a'])]
        resps_b = [str(row['response_b'])]

    # Setup Constants & Overhead
    # The instruction prompt
    instruction_text = (
        "\n# **Your task is to pick the best response between Model A and Model B or declare a Tie. "
        "Answer only with 'A', 'B', or 'Tie'. Think carefully before answering.**"
    )
    instruction_ids = tokenizer(instruction_text, add_special_tokens=False)['input_ids']
    
    # We leave a small buffer for safety
    current_budget = MAX_LENGTH - len(instruction_ids) - 10 
    
    formatted_rounds_ids = []
    
    # Zip the rounds together
    turns = list(zip(prompts, resps_a, resps_b))
    total_rounds = len(turns)

    # Iterate Backwards (Latest -> Earliest)
    # We prioritize the end of the conversation
    for i, (p_text, a_text, b_text) in enumerate(reversed(turns)):
        round_idx = total_rounds - i
        
        # Construct header
        header_text = f"\n\n## Round {round_idx}:\n"
        
        # Tokenize components without special tokens to get accurate counts
        header_ids = tokenizer(header_text, add_special_tokens=False)['input_ids']
        prompt_ids = tokenizer(f"### Prompt:\n{p_text}", add_special_tokens=False)['input_ids']
        resp_a_ids = tokenizer(f"\n### Response Model A:\n{a_text}", add_special_tokens=False)['input_ids']
        resp_b_ids = tokenizer(f"\n### Response Model B:\n{b_text}", add_special_tokens=False)['input_ids']
        
        # Calculate size of this specific round
        round_total_len = len(header_ids) + len(prompt_ids) + len(resp_a_ids) + len(resp_b_ids)
        
        # if the round fits the budget
        if round_total_len <= current_budget:
            # Add the full round
            full_round = header_ids + prompt_ids + resp_a_ids + resp_b_ids
            formatted_rounds_ids.insert(0, full_round) # Insert at front (because we are iterating backwards)
            current_budget -= round_total_len
        
        else:
            # This round is too big (or the budget is getting tight)
            # We apply sandwich logic to this round only to make it fit
            
            # space is left for the actual content
            available_for_content = current_budget - len(header_ids)
            
            # If we have practically no space left (<50 tokens), it's better to stop here 
            # than to include a meaningless fragment of a round
            if available_for_content < 50:
                break
                
            # Distribute budget: 20% Prompt, 40% Resp A, 40% Resp B
            p_limit = int(available_for_content * PROMPT_RATIO)
            a_limit = int(available_for_content * RESP_RATIO)
            b_limit = int(available_for_content * RESP_RATIO)
            
            # Truncate logic
            # Prompt: Keep START (Context/Intent)
            p_ids_cut = prompt_ids[:p_limit]
            
            # Responses: Keep END (Conclusion/Result)
            # Checking length prevents index errors if response is shorter than limit
            a_ids_cut = resp_a_ids[-a_limit:] if len(resp_a_ids) > a_limit else resp_a_ids
            b_ids_cut = resp_b_ids[-b_limit:] if len(resp_b_ids) > b_limit else resp_b_ids
            
            # Build the sandwiched round
            sandwiched_round = header_ids + p_ids_cut + a_ids_cut + b_ids_cut
            formatted_rounds_ids.insert(0, sandwiched_round)
            
            # Since we filled the budget, we stop processing earlier rounds (Left Truncation)
            break

    # Final Assembly
    # Flatten list of lists
    full_input_ids = [token for round_ids in formatted_rounds_ids for token in round_ids]
    
    # Append the mandatory instruction
    full_input_ids += instruction_ids
    
    # Decode back to text
    final_text = tokenizer.decode(full_input_ids, skip_special_tokens=True)
    
    # Add the "Look at..." prefix
    final_text = "# **Look at the following conversation history:** " + final_text
    
    return final_text

# def prepare_inference_input(row):
#     # Parse JSON strings to actual text
#     try:
#         prompt_text = "\n".join(json.loads(row['prompt']))
#         resp_a_text = "\n".join(json.loads(row['response_a']))
#         resp_b_text = "\n".join(json.loads(row['response_b']))
#     except (json.JSONDecodeError, TypeError):
#         prompt_text = str(row['prompt'])
#         resp_a_text = str(row['response_a'])
#         resp_b_text = str(row['response_b'])

#     # Tokenize to check lengths
#     prompt_ids = tokenizer(prompt_text, add_special_tokens=False)['input_ids']
#     resp_a_ids = tokenizer(resp_a_text, add_special_tokens=False)['input_ids']
#     resp_b_ids = tokenizer(resp_b_text, add_special_tokens=False)['input_ids']

#     # Apply Budget (Sandwich Logic)
#     max_prompt_len = int(AVAILABLE_TOKENS * PROMPT_RATIO)
#     max_resp_len = int(AVAILABLE_TOKENS * RESP_RATIO)

#     # Prompt: Keep Start
#     if len(prompt_ids) > max_prompt_len:
#         prompt_ids = prompt_ids[:max_prompt_len]

#     # Responses: Keep End
#     if len(resp_a_ids) > max_resp_len:
#         resp_a_ids = resp_a_ids[-max_resp_len:]

#     if len(resp_b_ids) > max_resp_len:
#         resp_b_ids = resp_b_ids[-max_resp_len:]

#     # Decode back to text
#     final_prompt = tokenizer.decode(prompt_ids, skip_special_tokens=True)
#     final_resp_a = tokenizer.decode(resp_a_ids, skip_special_tokens=True)
#     final_resp_b = tokenizer.decode(resp_b_ids, skip_special_tokens=True)

#     # Construct the Final Formatted String
#     return f"""# **Look at the following prompt and the two model responses:** \\n 
#     ## **Prompt:**
#     `{final_prompt}`\\n
#     ## **Response A:**
#     `{final_resp_a}`\\n
#     ## **Response B:**
#     `{final_resp_b}`
#     # **Your task is to pick the best response between Model A and Model B or declare a Tie. Answer only with 'A', 'B', or 'Tie'. Think carefully before answering.**"""



def prediction(formatted_text):
    # Ensure inputs don't exceed model limits even after manual formatting
    inputs = tokenizer(formatted_text,
                       return_tensors="pt",
                       truncation=True,
                       max_length=MAX_LENGTH).to("cuda:0")

    with torch.no_grad():
        outputs = wrapped_model(**inputs).logits

    # y_prob = np.round(np.array(torch.sigmoid(outputs).cpu().tolist()[0]), 5)
    # dim=-1 means normalize across the last dimension (the classes)
    probs_tensor = F.softmax(outputs, dim=-1)

    y_prob = np.round(np.array(probs_tensor.cpu().tolist()[0]), 5)

    # Sort labels by probability
    y_sorted_indices = np.argsort(y_prob)[::-1]
    y_sorted_labels = [id2class.get(y) for y in y_sorted_indices]
    y_prob_sorted = y_prob[y_sorted_indices]

    return y_sorted_labels, y_prob_sorted

# %% [markdown]
# With this function defined, we can now apply it to our test dataset. We’ll convert the test portion of dataset_toxic into a Pandas DataFrame for convenient processing. Then, using the .map() method, we'll apply our prediction function to each comment_text. The returned sorted labels and probabilities will be stored in new columns in our DataFrame.

# %%
# df_test = pd.read_csv("kaggle/input/lmsys-chatbot-arena/test.csv")

# df_test['pred'] = df_test['comment_text'].map(prediction)
# df_test['argsort_label']  = df_test['pred'].apply(lambda x : x[0])
# df_test['argsort_prob']   = df_test['pred'].apply(lambda x : x[1])
# print(df_test.shape)
# df_test.head(n=2)

# %%

# Load Test Data
df_test = pd.read_csv("kaggle/input/lmsys-chatbot-arena/test.csv")

print("Formatting inputs (Sandwich Strategy)...")
# Apply the formatting to every row (axis=1)
df_test['formatted_input'] = df_test.apply(prepare_inference_input, axis=1)

print("Running predictions...")
# Run prediction on the formatted text
df_test['pred'] = df_test['formatted_input'].map(prediction)

# Extract results
df_test['argsort_label'] = df_test['pred'].apply(lambda x: x[0])
df_test['argsort_prob'] = df_test['pred'].apply(lambda x: x[1])

# Clean up temporary columns
# df_test = df_test.drop(columns=['formatted_input', 'pred'])

print(df_test.shape)
df_test[['id', 'argsort_label', 'argsort_prob']].head(n=3)

# %% [markdown]
# Inspecting the first few rows of df_test (as shown in the example output table "Result of the dataframe" with "first two samples...") allows us to see the model's predictions directly alongside the original comments. For instance, looking at the second sample in such an output, we might observe that the 'toxic' category has a high probability (e.g., 0.95), while 'insult' and 'obscene' also show significant probabilities (e.g., around 0.8), and other categories have lower scores. This gives a direct insight into the model's assessment for each comment.
#
# This demonstrates a straightforward method to obtain and examine predictions. While various techniques exist for applying thresholds to these probabilities to derive final binary decisions for each label, this guide focuses on showcasing the raw predictive output for the test data, rather than delving into those specific post-processing strategies.

# %% [markdown]
# ## Saving the Fine-Tuned Model and LoRA Adapters
#
# Saving your fine-tuned model and its LoRA adapters correctly is crucial for future use, deployment, or sharing. The standard approach involves using the save_pretrained method available on your trained model object. This will save the adapter weights and an adapter configuration file.

# %%
output_dir = f'preference_class_gemma_1b_fix_2'
trainer.model.save_pretrained(output_dir)

# %% [markdown]
# While save_pretrained handles saving the adapter weights (usually in adapter_model.bin or .safetensors) and creates an adapter_config.json, special attention is needed for this configuration file. Given that we've made custom adjustments, particularly by potentially using a custom PEFT class wrapper like our Gemma3ForSequenceClassification, the automatically generated adapter_config.json might sometimes lack certain specific LoRA parameters or may not fully capture the nuances of our setup. An incomplete or misconfigured adapter_config.json can lead to difficulties when you later try to load the model with the PEFT library.
#
# To ensure this configuration file is robust and accurately reflects your setup, it’s prudent to programmatically verify and augment it. The following script demonstrates how you can load the saved adapter_config.json, compare it against your original LoraConfig object (which we'll refer to as lora_config from your setup) and the base model's Hugging Face ID, and then update any missing or incorrect fields.

# %%
# Assuming lora_config is the LoraConfig object used during setup
# Assuming hugging_face_model_id is the string ID like "google/gemma-3-4b-it"
# Assuming output_dir is the path where the model was saved

adapter_config_path = os.path.join(output_dir, "adapter_config.json")

# Check if file exists before proceeding
if os.path.exists(adapter_config_path):
    try:
        # Load the potentially incomplete config
        with open(adapter_config_path, 'r') as f:
            saved_config_dict = json.load(f)

        # Get parameters from the original LoraConfig
        # Use.to_dict() if available, otherwise __dict__
        try:
            # Ensure lora_config is the actual LoraConfig object instance
            lora_config_dict = lora_config.to_dict()
        except AttributeError:
            # Fallback, might include extra internal attributes
            lora_config_dict = lora_config.__dict__
            # Clean up potential internal attributes if using __dict__
            lora_config_dict = {
                k: v for k, v in lora_config_dict.items() if not k.startswith('_')}

        # *** FIX 1: Define the specific keys to check ***
        # These are common LoRA parameters that might be missing
        lora_keys_to_check = [
            "r",
            "lora_alpha",
            "lora_dropout",
            "target_modules",
            "bias",
            "modules_to_save",  # Important if you used it
            "fan_in_fan_out",
            "init_lora_weights",
            # Add any other specific keys from your LoraConfig if needed
        ]

        # Merge missing or None parameters from the original lora_config
        updated = False
        for key in lora_keys_to_check:
            # Check if key is missing in saved config OR if it exists but is None
            if key not in saved_config_dict or saved_config_dict[key] is None:
                # Check if the key exists in the original config and has a value
                if key in lora_config_dict and lora_config_dict[key] is not None:
                    saved_config_dict[key] = lora_config_dict[key]
                    updated = True

        # Ensure essential base fields are present and correct
        # Use getattr for safer access to lora_config attributes
        original_task_type = getattr(lora_config, 'task_type', 'SEQ_CLS')
        if 'task_type' not in saved_config_dict or saved_config_dict['task_type'] != original_task_type:
            saved_config_dict['task_type'] = original_task_type
            updated = True

        original_base_model = getattr(
            lora_config, 'base_model_name_or_path', hugging_face_model_id)
        if 'base_model_name_or_path' not in saved_config_dict or saved_config_dict['base_model_name_or_path'] != original_base_model:
            saved_config_dict['base_model_name_or_path'] = original_base_model
            updated = True

        if 'peft_type' not in saved_config_dict or saved_config_dict['peft_type'] != "LORA":
            saved_config_dict['peft_type'] = "LORA"
            updated = True

        # *** FIX 2: Convert set to list before saving ***
        if 'target_modules' in saved_config_dict and isinstance(saved_config_dict['target_modules'], set):
            saved_config_dict['target_modules'] = sorted(
                list(saved_config_dict['target_modules']))  # Convert set to sorted list
            updated = True  # Mark as updated if conversion happened

        if 'modules_to_save' in saved_config_dict and isinstance(saved_config_dict['modules_to_save'], set):
            # Also handle modules_to_save if it could be a set
            saved_config_dict['modules_to_save'] = sorted(
                list(saved_config_dict['modules_to_save']))
            updated = True

        # Overwrite the config file only if changes were made
        if updated:
            with open(adapter_config_path, 'w') as f:
                # Save the corrected dictionary as JSON
                json.dump(saved_config_dict, f, indent=2)
            print(
                f"Manually updated adapter configuration: {adapter_config_path}")
            # Optional: print the final dict
            print("New content:", saved_config_dict)
        else:
            print(
                f"Adapter configuration already seemed complete or no changes needed: {adapter_config_path}")

    except Exception as e:
        print(f"Error during manual update of adapter_config.json: {e}")
else:
    print(f"Error: adapter_config.json not found at {adapter_config_path}")

# %% [markdown]
# After running this script, if any modifications were needed, your adapter_config.json will be updated. This manual verification step helps ensure that all relevant details of your LoRA setup are accurately stored, which is key for reliably loading and reusing your fine-tuned adapter model with PEFT at a later stage.

# %% [markdown]
# ## Reloading the Saved Model and Making Predictions
#
# To use your fine-tuned model later, you’ll need to reload it. This involves setting up the base model architecture again, including any custom classes like our Gemma3ForSequenceClassification, and then loading the saved LoRA adapter weights. Ensure all necessary libraries, your custom class definitions, and configurations like class2id are available in your environment.
#
# First, we re-initialize the tokenizer and the base Gemma3ForCausalLM model with the same quantization settings (bnb_config) used during training. The model's language modeling head is then replaced with a new linear layer matching the number of labels for our classification task. We also re-establish the LoRA configuration that defines how adapters are applied. The base model, now with its classification head, is then wrapped using our Gemma3ForSequenceClassification class. This prepared structure is crucial for correctly loading and interpreting the saved adapter weights.
#
# With this setup in place, PeftModel.from_pretrained is used to load the trained LoRA adapter weights from your specified output_dir into the Gemma3ForSequenceClassification instance. We set is_trainable=False and switch the model to evaluation mode with model.eval()

# %%
output_dir = 'preference_class_gemma_1b_fix_2'
hugging_face_model_id = "google/gemma-3-1b-it"  # gemma-3-4b-it
gpu_device = 'cuda:0'


tokenizer = AutoTokenizer.from_pretrained(hugging_face_model_id,
                                          padding_side='right',
                                          device_map=gpu_device,
                                          add_bos=True)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

class2id = {'winner_model_a': 0, 'winner_model_b': 1, 'winner_tie': 2}
id2class = {v: k for k, v in class2id.items()}


bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16)

base_model = Gemma3ForCausalLM.from_pretrained(hugging_face_model_id,
                                               torch_dtype=torch.bfloat16,
                                               device_map=gpu_device,
                                               attn_implementation='flash_attention_2',  # eager
                                               quantization_config=bnb_config)

# %% [markdown]
# and Lora adapters

# %%

modules = ['gate_proj', 'down_proj', 'v_proj',
           'k_proj', 'q_proj', 'o_proj', 'up_proj']

lora_config = LoraConfig(
    r=64,
    lora_alpha=32,
    target_modules=modules,
    lora_dropout=0.1,
    bias="none",
    task_type="SEQ_CLS", modules_to_save=['lm_head'])

peft_config = PeftConfig(
    peft_type="LORA", task_type="SEQ_CLS", inference_mode=False)
for key, value in lora_config.__dict__.items():
    setattr(peft_config, key, value)


# Must match the number of classes used during training
num_labels = len(id2class.keys())
load_dtype = torch.bfloat16  # Match training or desired inference precision
print(f"Replacing lm_head for {num_labels} classes.")
base_model.lm_head = torch.nn.Linear(
    base_model.config.hidden_size,
    num_labels,
    bias=False,
    device=base_model.device  # Ensure head is on the correct device
).to(dtype=load_dtype)  # Ensure head matches model dtype


base_model = Gemma3ForSequenceClassification(peft_config, base_model)

# %% [markdown]
# load the saved model

# %%
model = PeftModel.from_pretrained(
    base_model,
    output_dir,
    is_trainable=False  # Set to False for inference
)

model.eval()

# %%
# def prediction(input_text):
#     inputs          = tokenizer(input_text, return_tensors="pt",).to("cuda:0")
#     with torch.no_grad():
#         outputs = model(**inputs).logits
#     y_prob          = np.round(np.array(torch.sigmoid(outputs).tolist()[0]),5)
#     y_sorted_labels = [id2class.get(y) for y  in np.argsort(y_prob)[::-1]]
#     y_prob_sorted   = np.sort(y_prob)[::-1]

#     return y_sorted_labels,y_prob_sorted

# %% [markdown]
# Example:

# %%


# %%
df_test

# %%
# example = ''' "Who the fuck are you?
# his fee was an umberella it was a joke made by himself i have sources let me post em up it was on SKY SPORTS NEWS.
# He was joking about the rain in manchester. So how the FUCK is that vandelising" '''
example = str(df_test.iloc[2]["formatted_input"])
prediction(example)

# %% [markdown]
# Using the prediction function with an example text demonstrates the reloaded model in action. The output shows the model's confidence scores for each category, sorted from most to least probable for the given input. This confirms that the saved adapters have been loaded correctly and the model is ready for inference tasks.

# %% [markdown]
# ## Conclusion
#
# This comprehensive guide has walked you through a practical example of fine-tuning a large language model using LoRA adapters, a powerful technique for efficient customization. While the principles of LoRA are broadly applicable, we specifically tackled the challenge of adapting the promising (though, for this article’s context, hypothetical) Gemma 3 model for a multi-label sequence classification task — a scenario where direct high-level Hugging Face classes might not yet exist.
#
# By demonstrating how to add a custom classification head, wrap the model within a tailored PEFT-compatible class, manually ensure the integrity of adapter configurations, and navigate the nuances of training and prediction, this article aimed to equip you with both the general methodology and specific strategies needed. The journey from loading a base causal model to making multi-label predictions showcases the flexibility and potential that emerges when combining foundational LLM capabilities with targeted adaptation techniques. We hope this detailed exploration serves as a valuable blueprint for your own projects, empowering you to fine-tune cutting-edge models like Gemma 3 for a diverse array of sequence classification challenges.
#
# Please consider this script and the accompanying guide as a foundational example, not an exhaustively optimized solution. Please feel free experiment further: adjust the LoRA adapter configurations, refine the custom Python classes, tweak the Trainer hyperparameters, or even customize the loss function—indeed, explore any component you see fit! There are numerous parameters and architectural choices that can be fine-tuned to potentially achieve even better performance. This has been a comprehensive demonstration designed to provide you with a solid starting point for your own explorations.
