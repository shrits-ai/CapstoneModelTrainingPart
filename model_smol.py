import torch
import torch.nn as nn
import math
from transformers.modeling_outputs import CausalLMOutputWithPast

# 1. Custom Configuration Class 
class CustomConfig:
    def __init__(self):
        # Architecture Parameters
        self.vocab_size = 49152
        self.hidden_size = 576          # d_model
        self.intermediate_size = 1536   # FFN dimension
        self.num_hidden_layers = 30     # Number of decoder layers
        self.num_attention_heads = 9    # Query heads
        self.num_key_value_heads = 3    # Key/Value heads
        self.max_position_embeddings = 2048
        self.rms_norm_eps = 1e-5
        self.rope_theta = 10000.0       # Rotary embedding base
        
        # Tokenizer/Generation Params
        self.pad_token_id = None
        self.bos_token_id = 0
        self.eos_token_id = 0
    
    def to_dict(self):
        # Serialize the config parameters
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}

# 2. Custom RMS Normalization
class CustomRMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        return self.weight * self._norm(x.float()).type_as(x)

# 3. Rotary Positional Embeddings
class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=2048, theta=10000.0):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self._set_cos_sin_cache(max_seq_len)

    def _set_cos_sin_cache(self, seq_len):
        t = torch.arange(seq_len, device=self.inv_freq.device)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :])
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :])

    def forward(self, x, seq_len):
        if seq_len > self.cos_cached.shape[2]:
            self._set_cos_sin_cache(seq_len)
        return self.cos_cached[:, :, :seq_len], self.sin_cached[:, :, :seq_len]

# 4. Attention Layer with Grouped Query Attention
class CustomAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_kv_heads = config.num_key_value_heads

        # Projections
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        # Rotary embeddings
        self.rotary_emb = RotaryEmbedding(
            self.head_dim,
            max_seq_len=config.max_position_embeddings,
            theta=config.rope_theta
        )

    def forward(self, x, attention_mask=None):
        batch_size, seq_len, _ = x.shape
        
        # Project queries/keys/values
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape for attention computation
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Apply rotary embeddings
        cos, sin = self.rotary_emb(x, seq_len=seq_len)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Repeat keys and values to match the number of query heads
        repeat_factor = self.num_heads // self.num_kv_heads
        k = k.repeat_interleave(repeat_factor, dim=1)
        v = v.repeat_interleave(repeat_factor, dim=1)

        # Attention scores
        attn_weights = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(self.head_dim)
        
        # Apply attention mask
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
            
        attn_weights = torch.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, v)

        # Reshape and project back
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.hidden_size)
        return self.o_proj(attn_output)

# 5. MLP Layer
class CustomMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        gate = self.act_fn(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)

# 6. Transformer Decoder Layer
class DecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attn = CustomAttention(config)
        self.mlp = CustomMLP(config)
        self.input_norm = CustomRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attn_norm = CustomRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, x, attention_mask=None):
        # Self-attention
        residual = x
        x = self.input_norm(x)
        x = self.self_attn(x, attention_mask)
        x = residual + x

        # MLP
        residual = x
        x = self.post_attn_norm(x)
        x = self.mlp(x)
        x = residual + x
        return x

# 7. Full Model
class CustomLLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([DecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = CustomRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lm_head.weight = self.embed_tokens.weight  # Tie the weights To reduce param

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids, attention_mask=None, labels=None):
        x = self.embed_tokens(input_ids)
        batch_size, seq_len = input_ids.shape
        
        # Create causal mask
        causal_mask = torch.full((seq_len, seq_len), float("-inf"), device=x.device)
        causal_mask = torch.triu(causal_mask, diagonal=1)
        causal_mask = causal_mask[None, None, :, :]  # Shape: [1, 1, seq_len, seq_len]
        
        # Combine with padding mask
        if attention_mask is not None:
            padding_mask = (1.0 - attention_mask.float()) * torch.finfo(x.dtype).min
            padding_mask = padding_mask.view(batch_size, 1, 1, seq_len)
            combined_mask = causal_mask + padding_mask
        else:
            combined_mask = causal_mask
        
        # Process through decoder layers
        for layer in self.layers:
            x = layer(x, attention_mask=combined_mask)
        
        x = self.norm(x)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            # Shift logits and labels for causal LM
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=None,
            hidden_states=None,
            attentions=None,
        )
    
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int = None,
        top_p: float = None,
        repetition_penalty: float = 1.0,
        eos_token_id: int = None,
        pad_token_id: int = None,
    ):
        """
        Generates text using various decoding strategies.
        
        Args:
            input_ids: Input token IDs of shape (batch_size, seq_len)
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_k: Top-k sampling cutoff
            top_p: Nucleus sampling cutoff
            repetition_penalty: Penalty for repeated tokens (1.0 = no penalty)
            eos_token_id: Stop generation when this token is produced
            pad_token_id: Padding token ID for sequence termination
            
        Returns:
            Generated sequence of token IDs
        """
        # Ensure model is in eval mode
        self.eval()
        
        # Move inputs to model device
        input_ids = input_ids.to(self.embed_tokens.weight.device)
        batch_size = input_ids.size(0)
        
        # Storage for generated sequences
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
        past_key_values = None  # Could implement KV caching here for efficiency
        
        for _ in range(max_new_tokens):
            # Forward pass (only compute last logits for efficiency)
            with torch.no_grad():
                outputs = self(input_ids)
                next_token_logits = outputs.logits[:, -1, :]

            # Repetition penalty
            if repetition_penalty != 1.0:
                next_token_logits = self._apply_repetition_penalty(
                    next_token_logits, input_ids, repetition_penalty
                )

            # Temperature scaling
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature

            # Top-k filtering
            if top_k is not None and top_k > 0:
                top_k_values, _ = torch.topk(next_token_logits, top_k)
                min_top_k = top_k_values[:, -1].unsqueeze(-1)
                next_token_logits = torch.where(
                    next_token_logits < min_top_k,
                    torch.tensor(-float('inf')).to(next_token_logits.device),
                    next_token_logits
                )

            # Top-p (nucleus) sampling
            if top_p is not None and top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_token_logits[indices_to_remove] = -float('inf')

            # Convert logits to probabilities
            probs = torch.softmax(next_token_logits, dim=-1)
            
            # Sample next tokens
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)

            # Update sequences
            input_ids = torch.cat([input_ids, next_tokens.unsqueeze(-1)], dim=-1)

            # Check for EOS tokens
            if eos_token_id is not None:
                unfinished = (next_tokens != eos_token_id).long() * unfinished_sequences
                unfinished_sequences = unfinished
                
                if unfinished_sequences.max() == 0:
                    break

        # Pad sequences if requested
        if pad_token_id is not None and eos_token_id is not None:
            input_ids = self._pad_sequences(input_ids, eos_token_id, pad_token_id)

        return input_ids

    def _apply_repetition_penalty(self, logits, sequences, penalty):
        """Applies repetition penalty to logits"""
        score = torch.gather(logits, 1, sequences)
        score = torch.where(score < 0, score * penalty, score / penalty)
        logits.scatter_(1, sequences, score)
        return logits

    def _pad_sequences(self, sequences, eos_token_id, pad_token_id):
        """Replace tokens after EOS with pad token"""
        # Create mask of positions after EOS
        eos_positions = (sequences == eos_token_id).int().argmax(dim=-1)
        padding_mask = torch.arange(sequences.size(1), device=sequences.device) > eos_positions.unsqueeze(-1)
        
        # Apply padding
        sequences[padding_mask] = pad_token_id
        return sequences

# Helper function for rotary embeddings
def apply_rotary_pos_emb(q, k, cos, sin):
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

'''
# Usage
config = CustomConfig()
model = CustomLLM(config)

# Verify parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params/1e6:.2f}M")  # Should output ~135.00M
print(model)
# Test forward pass after fix
input_ids = torch.randint(0, config.vocab_size, (1, 256))
output = model(input_ids)
print(output.shape)  # Expected output: (1, 256, 49152)

# Initialize model
config = CustomConfig()
model = CustomLLM(config)

# Generate text
prompt = torch.tensor([[config.bos_token_id]])  # Start token
generated = model.generate(
    prompt,
    max_new_tokens=50,
    temperature=0.7,
    top_p=0.9,
    eos_token_id=config.eos_token_id,
    pad_token_id=config.pad_token_id
)
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/cosmo2-tokenizer")
tokenizer.pad_token = tokenizer.eos_token  # For padding
# Decode tokens
generated_text = tokenizer.decode(generated[0].tolist())
print(prompt)
print(generated_text)
'''

