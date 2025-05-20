import  torch.nn.functional as F
import torch
from diffusers.models.attention_processor import Attention, AttentionProcessor
from torch.nn.attention.flex_attention import create_block_mask, flex_attention
create_block_mask = torch.compile(create_block_mask)
from typing import Optional
from functools import partial, lru_cache

@lru_cache
def init_local_mask_flex(frames, height, width, text_length, attenable_text, group_t, group_h, group_w, device):
    total_length=height*width*frames
    cell_size = group_t* group_h* group_w
    t,h,w = frames//group_t, height//group_h, width//group_w
    def local_mask(b, h_, q_idx, kv_idx):
        q_y=q_idx//cell_size
        kv_y = kv_idx//cell_size
        q_t = q_y//(h*w)
        q_h = (q_y%(h*w))//w
        q_w = (q_y%(h*w))%w
        
        kv_t = kv_y//(h*w)
        kv_h = (kv_y%(h*w))//w
        kv_w = (kv_y%(h*w))%w
        
        text = kv_idx<total_length+attenable_text
        text2 = torch.logical_or(q_idx>=total_length, torch.logical_and(kv_idx<total_length+attenable_text,  kv_idx>=total_length))

        image = torch.logical_and(torch.logical_or(torch.logical_or(q_t==kv_t, q_w==kv_w), q_h==kv_h), q_idx<total_length)
        
        return torch.logical_and(image | text2, text) 

    BLOCK_MASK = create_block_mask(local_mask, B=None, H=None, device=device,
                                   Q_LEN=text_length + height * width*frames,
                                   KV_LEN=text_length + height * width*frames, _compile=True)
    return BLOCK_MASK

class CrossAttnProcessor2_0:
    def __init__(self, mask, t, h, w, group_t, group_h, group_w, text_length):
        self.flex_attn = partial(flex_attention, block_mask=mask)
        self.flex_attn = torch.compile(self.flex_attn, dynamic=False)
        self.t, self.h, self.w = t,h,w
        self.group_t, self.group_h, self.group_w = group_t, group_h, group_w
        self.text_length=text_length
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "HunyuanVideoAttnProcessor2_0 requires PyTorch 2.0. To use it, please upgrade PyTorch to 2.0."
            )
    def clusterify(self, x):
        bsz, head, n, c = x.shape
        p_t, p_h, p_w=self.group_t, self.group_h, self.group_w
        t,h,w=self.t, self.h, self.w
        t_, h_, w_ = t//p_t, h // p_h, w // p_w
        x = x.reshape(bsz, head, t_, p_t, h_, p_h, w_, p_w, c)
        x = torch.einsum('nxtahpwqc->nxthwapqc', x)
        x = x.reshape(bsz, head, -1, c)
        return x
    def unclusterify(self, x):
        bsz, head, n, c = x.shape
        p_t, p_h, p_w = self.group_t, self.group_h, self.group_w
        t, h, w = self.t, self.h, self.w
        t_, h_, w_ = t//p_t, h // p_h, w // p_w
        x = x.reshape(bsz, head, t_, h_, w_, p_t, p_h, p_w, c)
        x = torch.einsum('nxthwapqc->nxtahpwqc', x)
        x = x.reshape(bsz, head, -1, c)
        return x

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if attn.add_q_proj is None and encoder_hidden_states is not None:
            hidden_states = torch.cat([hidden_states, encoder_hidden_states], dim=1)

        # 1. QKV projections
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)


        query = query.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        key = key.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        value = value.unflatten(2, (attn.heads, -1)).transpose(1, 2)



        # 2. QK normalization
        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # 3. Rotational positional embeddings applied to latent stream
        if image_rotary_emb is not None:
            from diffusers.models.embeddings import apply_rotary_emb

            if attn.add_q_proj is None and encoder_hidden_states is not None:
                query = torch.cat(
                    [
                        apply_rotary_emb(query[:, :, : -encoder_hidden_states.shape[1]], image_rotary_emb),
                        query[:, :, -encoder_hidden_states.shape[1] :],
                    ],
                    dim=2,
                )
                key = torch.cat(
                    [
                        apply_rotary_emb(key[:, :, : -encoder_hidden_states.shape[1]], image_rotary_emb),
                        key[:, :, -encoder_hidden_states.shape[1] :],
                    ],
                    dim=2,
                )
            else:
                query = apply_rotary_emb(query, image_rotary_emb)
                key = apply_rotary_emb(key, image_rotary_emb)

        # 4. Encoder condition QKV projection and normalization
        if attn.add_q_proj is not None and encoder_hidden_states is not None:
            encoder_query = attn.add_q_proj(encoder_hidden_states)
            encoder_key = attn.add_k_proj(encoder_hidden_states)
            encoder_value = attn.add_v_proj(encoder_hidden_states)

            encoder_query = encoder_query.unflatten(2, (attn.heads, -1)).transpose(1, 2)
            encoder_key = encoder_key.unflatten(2, (attn.heads, -1)).transpose(1, 2)
            encoder_value = encoder_value.unflatten(2, (attn.heads, -1)).transpose(1, 2)

            if attn.norm_added_q is not None:
                encoder_query = attn.norm_added_q(encoder_query)
            if attn.norm_added_k is not None:
                encoder_key = attn.norm_added_k(encoder_key)

            query = torch.cat([query, encoder_query], dim=2)
            key = torch.cat([key, encoder_key], dim=2)
            value = torch.cat([value, encoder_value], dim=2)
        
        query_image, query_text  = query[:, :, :-self.text_length], query[:, :, -self.text_length:]  #b h  n c
        key_image, key_text  = key[:, :, :-self.text_length], key[:, :, -self.text_length:]
        value_image, value_text  = value[:, :, :-self.text_length], value[:, :, -self.text_length:]

        query_image = self.clusterify(query_image)
        key_image = self.clusterify(key_image)
        value_image = self.clusterify(value_image)
        query = torch.cat([query_image, query_text, ],dim=2)
        key = torch.cat([key_image, key_text, ], dim=2)
        value = torch.cat([value_image, value_text, ], dim=2)

        hidden_states = self.flex_attn(query, key, value)
        hidden_states_image, hidden_states_text = hidden_states[:, :, :-self.text_length], hidden_states[:, :, -self.text_length:]  #b h  n c
        hidden_states_image = self.unclusterify(hidden_states_image)
        hidden_states = torch.cat([hidden_states_image, hidden_states_text],dim=2)
        hidden_states = hidden_states.transpose(1, 2).flatten(2, 3)
        hidden_states = hidden_states.to(query.dtype)


        # 6. Output projection
        if encoder_hidden_states is not None:
            hidden_states, encoder_hidden_states = (
                hidden_states[:, : -encoder_hidden_states.shape[1]],
                hidden_states[:, -encoder_hidden_states.shape[1] :],
            )

            if getattr(attn, "to_out", None) is not None:
                hidden_states = attn.to_out[0](hidden_states)
                hidden_states = attn.to_out[1](hidden_states)

            if getattr(attn, "to_add_out", None) is not None:
                encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

        return hidden_states, encoder_hidden_states