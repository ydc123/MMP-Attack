import os
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import StableDiffusionPipeline
from transformers import CLIPTokenizer, CLIPModel, CLIPProcessor
from PIL import Image
import json
from tqdm import tqdm
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--ori_sentence', type=str, default='a photo of car')
parser.add_argument('--target_word', type=str, default='bird')
parser.add_argument('--iteration', type=int, default=10000)
parser.add_argument('--num_new_token', type=int, default=4)
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--weight_text', type=float, default=0.1, help='weight for text loss')
args = parser.parse_args()

device = 'cuda'
clip_dir = 'models/clip-vit-large-patch14'
diff_dir = 'models/stable-diffusion-v1-4'
clip_model = CLIPModel.from_pretrained(clip_dir).to(device)
preprocess = CLIPProcessor.from_pretrained(clip_dir)
tokenizer = CLIPTokenizer.from_pretrained(clip_dir)
diff_model = StableDiffusionPipeline.from_pretrained(diff_dir, revision='fp16',
                                            torch_dtype=torch.float32, use_auth_token=True, safety_checker=None).to(device)
tar_sentence = f'a photo of {args.target_word}'

def get_text_embedding(inputs):
    '''
    inputs: a str or a tokenized Tensor (input_ids)
    '''
    if isinstance(inputs, str):
        input_ids = tokenizer(inputs, padding='max_length', max_length=tokenizer.model_max_length,
                            truncation=True, return_tensors='pt').input_ids.to(device)[0]
    else:
        input_ids = inputs                    
    input_ids = input_ids.unsqueeze(0)
    pooled_output = diff_model.text_encoder(input_ids)[1]
    proj_emb = clip_model.text_projection(pooled_output)
    return proj_emb


def get_ascii_toks(tokenizer, embed_weights, device, target_token):

    def is_ascii(s):
        return s.isascii() and s.isprintable()

    ascii_toks = []
    for i in range(3, tokenizer.vocab_size):
        if is_ascii(tokenizer.decoder[i]) and tokenizer.decoder[i].endswith('</w>'):
            if tokenizer.decoder[i][:-4].isalpha() == False:
                continue
            s1 = tokenizer.decode([i])
            s2 = tokenizer.decode(tokenizer.encode(s1), skip_special_tokens=True)
            if s1 == s2:
                ascii_toks.append(i)
    forbidden_tokens = []
    # remove the top-k most similar tokens
    weights_concept = embed_weights[target_token]
    cos = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)
    cosine_values = []
    for idx in ascii_toks:
        weights_idx = embed_weights[idx]
        cosine_values.append(cos(weights_concept, weights_idx))
    cosine_values = torch.tensor(cosine_values, device=device)
    _, topk = torch.topk(cosine_values, k=20, largest=True)
    # print('Following words are not allowed:')
    for idx in topk:
        forbidden_tokens.append(tokenizer.decode([ascii_toks[idx]]))
        # print(tokenizer.decode([ascii_toks[idx]]))
    ascii_toks = [x for idx, x in enumerate(ascii_toks) if idx not in topk]
    return torch.tensor(ascii_toks, device=device), forbidden_tokens

## load models

cos = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)
loss_fn = lambda x, y: 1 - cos(x.view(-1), y.view(-1))

tar_embs = []
target_token = tokenizer.encoder[args.target_word + '</w>']
image = Image.open(f'./reference_images/{args.target_word}.png')
with torch.no_grad():
    inputs = preprocess(images=[image], return_tensors="pt").to(device)
    tar_emb_image = clip_model.get_image_features(**inputs).detach()
    tar_emb_text = get_text_embedding(tar_sentence).detach()
input_ids = tokenizer(
    args.ori_sentence, padding='max_length', max_length=tokenizer.model_max_length,
    truncation=True, return_tensors='pt').input_ids.to(device)[0]
for idx in range(input_ids.shape[0]):
    if input_ids[idx] == tokenizer.eos_token_id:
        pos_eos = idx
        break
slice_adv = range(pos_eos, pos_eos + args.num_new_token)

embed_weights = diff_model.text_encoder.get_input_embeddings().weight.data
allowed_tokens, forbidden_tokens = get_ascii_toks(tokenizer, embed_weights, device, target_token)
print('Following words are not allowed:')
for token in forbidden_tokens:
    print(token)
not_allowed_tokens = set(list(range(tokenizer.vocab_size))) - set(allowed_tokens.tolist())
not_allowed_tokens = torch.tensor(list(not_allowed_tokens), device=device)

best_decode_str = None
best_input_ids = None
adv_emb = get_text_embedding(input_ids)
loss_original = 0
print('=' * 100)

print(target_token, tokenizer.decoder[target_token], input_ids[slice_adv])
target_token_embedding = embed_weights[target_token]
cos = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)
max_sim = -1e9
for idx in allowed_tokens:
    cos_value = cos(embed_weights[idx], target_token_embedding)
    if max_sim < cos_value:
        max_sim = cos_value
        adv_token = torch.tensor([idx] * args.num_new_token, device=device)
print(adv_token, tokenizer.decode(adv_token))
input_ids[pos_eos + args.num_new_token] = tokenizer.eos_token_id # for SD 2.1

adv_token_embed = embed_weights[adv_token].detach().requires_grad_(True)
embed_weights = embed_weights[allowed_tokens]
optim = torch.optim.Adam([adv_token_embed], lr=args.lr)
text_model = diff_model.text_encoder.text_model
input_embed = text_model.embeddings.token_embedding(input_ids).detach()
best_loss = 1e100
for i in tqdm(range(args.iteration)):
    # quantize adv_token with a quantize table "embed_weights"
    with torch.no_grad():
        diff = torch.sum((adv_token_embed.data.unsqueeze(1) - embed_weights.unsqueeze(0)) ** 2, dim=-1)
        token_idx = diff.argmin(dim=1)
        q_adv_token_embed = embed_weights[token_idx]
    q_adv_token_embed = q_adv_token_embed.data - adv_token_embed.data + adv_token_embed
    # q_adv_token_embed = adv_token_embed

    full_embed = torch.cat([input_embed[:pos_eos, :], q_adv_token_embed, input_embed[pos_eos + args.num_new_token:, :]], dim=0)
    # refer to https://github.com/huggingface/transformers/blob/v4.26.0/src/transformers/models/clip/modeling_clip.py#L687
    # refer to https://github.com/huggingface/transformers/blob/v4.26.0/src/transformers/models/clip/modeling_clip.py#L215

    output_attentions = text_model.config.output_attentions
    output_hidden_states = (
        text_model.config.output_hidden_states
    )
    return_dict = text_model.config.use_return_dict

    hidden_states = text_model.embeddings(inputs_embeds=full_embed)

    bsz, seq_len = 1, input_ids.shape[0]
    # CLIP's text model uses causal mask, prepare it here.
    # https://github.com/openai/CLIP/blob/cfcffb90e69f37bf2ff1e988237a0fbe41f33c04/clip/model.py#L324
    causal_attention_mask = text_model._build_causal_attention_mask(bsz, seq_len, hidden_states.dtype).to(
        hidden_states.device
    )
    # expand attention_mask

    encoder_outputs = text_model.encoder(
        inputs_embeds=hidden_states,
        causal_attention_mask=causal_attention_mask,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )

    last_hidden_state = encoder_outputs[0]
    last_hidden_state = text_model.final_layer_norm(last_hidden_state)
    pooled_output = last_hidden_state[
        torch.arange(adv_emb.shape[0], device=adv_emb.device),
        torch.tensor(pos_eos + args.num_new_token),
    ]
    adv_emb = clip_model.text_projection(pooled_output)
    loss = loss_fn(adv_emb, tar_emb_text) * args.weight_text + loss_fn(adv_emb, tar_emb_image)
    cur_input_ids = input_ids.clone()
    for idx in slice_adv:
        cur_input_ids[idx] = allowed_tokens[token_idx[idx - pos_eos]]
    if best_loss > loss.item():
        best_loss = loss.item()
        best_result = tokenizer.decode(cur_input_ids, skip_special_tokens=True)
    optim.zero_grad()
    loss.backward()
    optim.step()
print(best_result)
