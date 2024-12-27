import os
import sys

with open(sys.argv[0]) as f:
    code = f.read()  # read the code of this file ASAP, for logging
import uuid
import time
import contextlib
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
import torch._inductor.config as config
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.attention.flex_attention import BlockMask, flex_attention  # KoszarskyB


# -----------------------------------------------------------------------------
# PyTorch nn.Module definitions for the GPT-2 model


def norm(x):
    return F.rms_norm(x, (x.size(-1),))


class CastedLinear(nn.Linear):

    def __init__(self, in_features, out_features):
        super().__init__(in_features, out_features, bias=False)

    def forward(self, x):
        return F.linear(x, self.weight.to(x.dtype))


class Rotary(torch.nn.Module):

    def __init__(self, dim, base=10000):
        super().__init__()
        self.register_buffer("inv_freq", (1 / base) ** (torch.arange(0, dim, 2) / dim))
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x):
        seq_len = x.shape[1]
        if seq_len != self.seq_len_cached:
            t = torch.arange(seq_len, device=x.device)
            freqs = torch.outer(t, self.inv_freq)
            self.seq_len_cached = seq_len
            self.cos_cached = freqs.cos()
            self.sin_cached = freqs.sin()
        cos, sin = self.cos_cached[None, :, None, :], self.sin_cached[None, :, None, :]
        # apply_rotary_emb(x, cos, sin)
        x1, x2 = x.chunk(2, dim=3)
        y1 = x1 * cos + x2 * sin
        y2 = x1 * (-sin) + x2 * cos
        return torch.cat((y1, y2), 3).type_as(x)


class CausalSelfAttention(nn.Module):

    def __init__(self, dim, num_heads):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.c_q = CastedLinear(dim, dim)
        self.c_k = CastedLinear(dim, dim)
        self.c_v = CastedLinear(dim, dim)
        self.lambdas = nn.Parameter(torch.tensor([0.5, 0.5]))
        self.rotary = Rotary(dim // num_heads)  # dim // num_heads = head_dim
        self.c_proj = CastedLinear(dim, dim)
        self.c_proj.weight.data.zero_()  # zero init suggested by @Grad62304977

    def forward(self, x, vi, block_mask):
        B, T = x.size(0), x.size(1)  # batch size, sequence length
        assert B == 1, "Must use batch size = 1 for FlexAttention"
        q = self.c_q(x).view(B, T, self.num_heads, -1)
        k = self.c_k(x).view(B, T, self.num_heads, -1)
        v = self.c_v(x).view(B, T, self.num_heads, -1)
        v = self.lambdas[0] * v + self.lambdas[1] * vi.view_as(
            v
        )  # @KoszarskyB & @Grad62304977
        q, k = norm(q), norm(k)  # QK norm @Grad62304977
        q, k = self.rotary(q), self.rotary(k)
        y = flex_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            block_mask=block_mask,
            enable_gqa=True,
        )
        y = (
            y.transpose(1, 2).contiguous().view_as(x)
        )  # re-assemble all head outputs side by side
        y = self.c_proj(y)
        return y


class MLP(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.c_fc = CastedLinear(dim, 4 * dim)
        self.c_proj = CastedLinear(4 * dim, dim)
        self.c_proj.weight.data.zero_()  # zero init suggested by @Grad62304977

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(
            x
        ).square()  # https://arxiv.org/abs/2109.08668v2; ~1-2% better than GELU; suggested by @SKYLINEZ007 and @Grad62304977
        x = self.c_proj(x)
        return x


class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.attn = CausalSelfAttention(config.model_dim, config.num_heads)
        self.mlp = MLP(config.model_dim)
        self.lambdas = nn.Parameter(torch.tensor([1.0, 0.0]))

    def forward(self, x, vi, x0, block_mask):
        x = self.lambdas[0] * x + self.lambdas[1] * x0
        x = x + self.attn(norm(x), vi, block_mask)
        x = x + self.mlp(norm(x))
        return x


class ValueEmbedding(nn.Module):
    def __init__(self, config: "GPTConfig"):
        super().__init__()
        self.embed = nn.ModuleList(
            [nn.Embedding(config.vocab_size, config.model_dim) for _ in range(6)]
        )

    def forward(self, inputs) -> "list[torch.Tensor]":
        ve = [emb(inputs) for emb in self.embed]
        ve += reversed(ve)
        return ve


# -----------------------------------------------------------------------------
# The main GPT-2 model


@dataclass
class GPTConfig:
    vocab_size: int = 50304
    num_layers: int = 12
    num_heads: int = 6  # head dim 128 suggested by @Grad62304977
    model_dim: int = 768


class GPT(nn.Module):

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.num_layers = config.num_layers

        # U-net design by @brendanh0gan
        self.num_encoder_layers = (
            config.num_layers // 2
        )  # Half of the layers for encoder
        self.num_decoder_layers = (
            config.num_layers - self.num_encoder_layers
        )  # Remaining for decoder
        # Add learnable skip connection weights for decoder layers
        self.skip_weights = nn.Parameter(torch.ones(self.num_decoder_layers))

        self.embed = nn.Embedding(config.vocab_size, config.model_dim)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.num_layers)])
        # token value embeddings by @KoszarskyB - inspired by @Grad62304977's value residual learning
        # U-net structure on token value embeddings by @leloykun
        self.value_embeds = ValueEmbedding(config)
        self.lm_head = CastedLinear(config.model_dim, config.vocab_size)
        self.lm_head.weight.data.zero_()  # @Grad62304977

    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        sliding_window_num_blocks: torch.Tensor,
    ):
        BLOCK_SIZE = 128
        seq_len = len(inputs)
        assert seq_len % BLOCK_SIZE == 0
        total_num_blocks = seq_len // BLOCK_SIZE
        assert inputs.ndim == 1
        docs = (inputs == 50256).cumsum(0)
        docs_low = docs.view(-1, BLOCK_SIZE)[:, 0].contiguous()
        docs_high = docs.view(-1, BLOCK_SIZE)[:, -1].contiguous()

        def document_causal(b, h, q_idx, kv_idx):
            causal_mask = q_idx >= kv_idx
            document_mask = docs[q_idx] == docs[kv_idx]
            return causal_mask & document_mask

        def dense_to_ordered(dense_mask: torch.Tensor):
            num_blocks = dense_mask.sum(dim=-1, dtype=torch.int32)
            indices = dense_mask.argsort(dim=-1, descending=True, stable=True).to(
                torch.int32
            )
            return num_blocks[None, None].contiguous(), indices[None, None].contiguous()

        def create_doc_swc_block_mask(sliding_window_num_blocks: torch.Tensor):
            kv_idx = block_idx = torch.arange(
                total_num_blocks, dtype=torch.int32, device="cuda"
            )
            q_idx = block_idx[:, None]
            causal_bm = q_idx >= kv_idx
            causal_full_bm = q_idx > kv_idx
            window_bm = q_idx - kv_idx < sliding_window_num_blocks
            window_full_bm = window_bm
            # document_bm = (docs_low[q_idx] <= docs_high[kv_idx]) & (docs_low[kv_idx] <= docs_high[q_idx])
            document_bm = (docs_low[:, None] <= docs_high) & (
                docs_low <= docs_high[:, None]
            )
            document_full_bm = (docs_low[:, None] == docs_high) & (
                docs_low == docs_high[:, None]
            )
            nonzero_bm = causal_bm & window_bm & document_bm
            full_bm = causal_full_bm & window_full_bm & document_full_bm
            kv_num_blocks, kv_indices = dense_to_ordered(nonzero_bm ^ full_bm)
            full_kv_num_blocks, full_kv_indices = dense_to_ordered(full_bm)
            return BlockMask.from_kv_blocks(
                kv_num_blocks,
                kv_indices,
                full_kv_num_blocks,
                full_kv_indices,
                BLOCK_SIZE=BLOCK_SIZE,
                mask_mod=document_causal,
            )

        block_mask = create_doc_swc_block_mask(sliding_window_num_blocks)

        # forward the GPT model itself
        x = self.embed(inputs[None])  # token embeddings of shape (b, t, model_dim)
        x = norm(x)  # @Grad62304977
        x0 = x
        ve = self.value_embeds(inputs)
        ve_enc, ve_dec = ve[: self.num_encoder_layers], ve[self.num_encoder_layers :]

        # Store outputs for U-Net skip connections
        skip_connections = []
        # Encoder pass - process only the first half of the blocks
        for i in range(self.num_encoder_layers):
            x = self.blocks[i](x, ve_enc[i], x0, block_mask)
            skip_connections.append(x)
        # Decoder pass - process the remaining blocks with weighted skip connections
        for i in range(self.num_decoder_layers):
            x = x + self.skip_weights[i] * skip_connections.pop()
            # U-net structure on token value embeddings by @leloykun
            x = self.blocks[self.num_encoder_layers + i](x, ve_dec[i], x0, block_mask)

        x = norm(x)
        logits = self.lm_head(x)
        logits = 30 * torch.tanh(logits / 30)  # @Grad62304977
        logits = logits.float()
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return loss


# -----------------------------------------------------------------------------
# Our own simple Distributed Data Loader


def _peek_data_shard(file: Path):
    # only reads the header, returns header data
    # header is 256 int32
    header = torch.from_file(f"{file}", False, 256, dtype=torch.int32)
    assert header[0] == 20240520, "magic number mismatch in the data .bin file"
    assert header[1] == 1, "unsupported version"
    return int(header[2])  # number of tokens (claimed)


def _load_data_shard(path: Path, num_tokens):
    with path.open("rb", buffering=0) as f:
        tokens = torch.empty(num_tokens, dtype=torch.uint16, pin_memory=True)
        f.seek(256 * 4)
        nbytes = f.readinto(tokens.numpy())
        assert nbytes == 2 * num_tokens, "number of tokens read does not match header?"
    return tokens


class DistributedDataLoader:
    def __init__(self, filename_pattern, seq_len, process_rank, num_processes):
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.seq_len = seq_len

        # glob files that match the pattern
        self.files = sorted(Path.cwd().glob(filename_pattern))
        assert (
            len(self.files) > 0
        ), f"did not find any files that match the pattern {filename_pattern}"

        # load and validate all data shards, count number of tokens in total
        self.files_num_tokens = [_peek_data_shard(file) for file in self.files]
        assert min(self.files_num_tokens) >= num_processes * seq_len + 1
        self.total_num_tokens = sum(self.files_num_tokens)

        self.reset()

    def reset(self):
        self.current_shard = -1
        self.advance()

    def advance(self):  # advance to next data shard
        self.current_shard = (self.current_shard + 1) % len(self.files)
        self.current_position = self.process_rank * self.seq_len
        self.tokens = _load_data_shard(
            self.files[self.current_shard], self.files_num_tokens[self.current_shard]
        )

    def next_batch(self):
        batch_size = self.seq_len * self.num_processes
        buf = self.tokens[
            self.current_position : self.current_position + self.seq_len + 1
        ]
        # host side async is sufficient;
        # no performance improvement was observed when introducing a separate stream.
        inputs = buf[:-1].to(
            device="cuda", dtype=torch.int32, non_blocking=True
        )  # inputs
        targets = buf[1:].to(
            device="cuda", dtype=torch.int64, non_blocking=True
        )  # targets
        # advance current position and load next shard if necessary
        self.current_position += batch_size
        if self.current_position + batch_size + 1 >= len(self.tokens):
            self.advance()
        return inputs, targets


# -----------------------------------------------------------------------------
# Training setup and main loop


@dataclass
class Hyperparameters:
    # data hyperparams
    input_bin: str = "data/fineweb10B/fineweb_train_*.bin"  # input .bin to train on
    input_val_bin: str = (
        "data/fineweb10B/fineweb_val_*.bin"  # input .bin to eval validation loss on
    )
    # optimization hyperparams
    batch_size: int = 8  # batch size, in sequences, across all devices
    sequence_length: int = 64 * 1024  # sequence length, in tokens
    num_iterations: int = 1480  # number of iterations to run
    warmup_iters: int = 0
    cooldown_iters: int = (
        600  # number of iterations of linear warmup/cooldown for triangular or trapezoidal schedule
    )
    weight_decay: float = 0
    # evaluation and logging hyperparams
    val_loss_every: int = (
        125  # every how many steps to evaluate val loss? 0 for only at the end
    )
    val_tokens: int = (
        10485760  # how many tokens of validation data? it's important to keep this fixed for consistent comparisons
    )


HParams = Hyperparameters
args = HParams()

# set up DDP (distributed data parallel). torchrun sets this env variable
ddp_rank = int(os.environ["RANK"])
ddp_local_rank = int(os.environ["LOCAL_RANK"])
ddp_world_size = int(os.environ["WORLD_SIZE"])
assert torch.cuda.is_available()
device = torch.device(f"cuda:{ddp_local_rank}")
torch.cuda.set_device(device)
print(f"using device: {device}")
dist.init_process_group(backend="nccl", device_id=device)
dist.barrier()
master_process = ddp_rank == 0  # this process will do logging, checkpointing etc.

# begin logging
logfile = None
if master_process:
    run_id = uuid.uuid4()
    Path("logs").mkdir(exist_ok=True)
    # logdir = Path('logs') / f'{run_id}'
    # logdir.mkdir()
    logfile = Path("logs") / f"{run_id}.txt"
    print(logfile.stem)
    # create the log file
    with logfile.open("w") as f:
        # begin the log by printing this file (the Python code)
        print(code, file=f)
        print("=" * 100, file=f)


def print0(s, logonly=False):
    if master_process:
        with logfile.open("a") as f:
            if not logonly:
                print(s)
            print(s, file=f)


# log information about the hardware/software environment this is running on
# and print the full `nvidia-smi` to file
print0(f"Running python {sys.version}")
print0(
    f"Running pytorch {torch.version.__version__} compiled for CUDA {torch.version.cuda}\nnvidia-smi:"
)
import subprocess

result = subprocess.run(
    ["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
)
print0(f"{result.stdout}", logonly=True)
print0("=" * 100, logonly=True)

# calculate the number of steps to take in the val loop.
assert args.val_tokens % (args.sequence_length * ddp_world_size) == 0
val_steps = args.val_tokens // (args.sequence_length * ddp_world_size)
# calculate the steps of gradient accumulation required to attain the desired global batch size.
assert args.batch_size % (ddp_world_size) == 0
train_accumulation_steps = args.batch_size // ddp_world_size

# load tokens
train_loader = DistributedDataLoader(
    args.input_bin, args.sequence_length, ddp_rank, ddp_world_size
)
val_loader = DistributedDataLoader(
    args.input_val_bin, args.sequence_length, ddp_rank, ddp_world_size
)
print0(
    f"Training DataLoader: total number of tokens: {train_loader.total_num_tokens} across {len(train_loader.files)} files"
)
print0(
    f"Validation DataLoader: total number of tokens: {val_loader.total_num_tokens} across {len(val_loader.files)} files"
)
print0("=" * 100, logonly=True)
inputs_train, targets_train = train_loader.next_batch()

# there are only 50257 unique GPT-2 tokens; we extend to nearest multiple of 128 for efficiency. suggested to me by @Grad62304977.
# this originates from Karpathy's experiments.
num_vocab = 50304
model = GPT(GPTConfig(vocab_size=num_vocab, num_layers=12, num_heads=6, model_dim=768))
model = model.cuda().bfloat16()
for m in model.modules():
    if isinstance(m, CastedLinear):
        m.float()
config.coordinate_descent_tuning = True  # suggested by @Chillee
model = torch.compile(model)
# here we wrap model into DDP container
model = DDP(
    model,
    device_ids=[ddp_local_rank],
    broadcast_buffers=False,
    gradient_as_bucket_view=True,
)
raw_model = model.module  # always contains the "raw" unwrapped model

# init the optimizer
optimizer = torch.optim.Adam(
    model.parameters(), lr=0.0003, betas=(0.9, 0.95), fused=True
)


# learning rate decay scheduler (linear warmup and cooldown)
def get_lr(it):
    assert it <= args.num_iterations
    # 1) linear warmup for warmup_iters steps
    if it < args.warmup_iters:
        return (it + 1) / args.warmup_iters
    # 2) constant lr for a while
    elif it < args.num_iterations - args.cooldown_iters:
        return 1.0
    # 3) linear cooldown
    else:
        decay_ratio = (args.num_iterations - it) / args.cooldown_iters
        return decay_ratio


scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, get_lr)

sliding_window_num_blocks = torch.tensor(1, dtype=torch.int32, device="cuda")
sw_num_blocks_prev = 1
# Start training loop
training_time_ms = 0
# start the clock
torch.cuda.synchronize()
t0 = time.perf_counter()
# begin training
for step in range(args.num_iterations + 1):
    last_step = step == args.num_iterations
    # This effectively ignores timing first 10 steps, which are slower for weird reasons.
    # Alternately, and slightly more correctly in terms of benchmarking, we could do 10
    # steps with dummy data first, and then re-initialize the model and reset the loader.
    if step == 10:
        training_time_ms = 0
        t0 = time.perf_counter()
    timed_steps = (
        float("nan") if step <= 11 else (step - 10) + 1
    )  # <= 11 to avoid bug in val

    # Linearly increase the sliding window size over training in chunks of 128 from 128 -> 1856. By @fernbear.bsky.social
    frac_done = step / args.num_iterations  # training progress
    sw_num_blocks = int(((1 - frac_done) * 128 + frac_done * 1856) // 128)
    if sw_num_blocks != sw_num_blocks_prev:
        sliding_window_num_blocks.copy_(sw_num_blocks, non_blocking=True)
        sw_num_blocks_prev = sw_num_blocks

    # once in a while evaluate the validation dataset
    if last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0):
        # stop the clock
        torch.cuda.synchronize()
        training_time_ms += 1000 * (time.perf_counter() - t0)
        # run validation batches
        model.eval()
        val_loader.reset()
        val_loss = 0.0
        for _ in range(val_steps):
            with torch.no_grad():
                inputs_val, targets_val = val_loader.next_batch()
                val_loss += model(inputs_val, targets_val, sliding_window_num_blocks)
        dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
        val_loss /= val_steps
        # log val loss to console and to logfile
        print0(
            f"step:{step}/{args.num_iterations} val_loss:{val_loss:.4f} train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms/(timed_steps-1):.2f}ms"
        )
        # start the clock again
        torch.cuda.synchronize()
        t0 = time.perf_counter()

    # uncomment if you want to save any checkpoints
    # save_every = 1000
    # if master_process and (last_step or (save_every > 0 and step % save_every == 0)):
    #    # stop the clock
    #    torch.cuda.synchronize()
    #    training_time_ms += 1000 * (time.perf_counter() - t0)
    #    # save the state of the training process
    #    log = dict(step=step, code=code, model=raw_model.state_dict(), optimizer=optimizer.state_dict())
    #    torch.save(log, 'logs/%s/state_step%06d.pt' % (run_id, step))
    #    # start the clock again
    #    torch.cuda.synchronize()
    #    t0 = time.perf_counter()

    # bit confusing: we want to make sure to eval on 0th iteration
    # but also after the very last iteration. so we loop for step <= num_iterations
    # instead of just < num_iterations (one extra due to <=), only to do
    # the validation/sampling one last time, and then we break right here as we're done.
    if last_step:
        break

    # --------------- TRAINING SECTION BEGIN -----------------
    model.train()
    for i in range(1, train_accumulation_steps + 1):
        with contextlib.ExitStack() as stack:
            if (
                i < train_accumulation_steps
            ):  # there's no need to sync gradients every accumulation step
                stack.enter_context(model.no_sync())
            if step >= 5:
                stack.enter_context(
                    torch.compiler.set_stance(skip_guard_eval_unsafe=True)
                )
            model(inputs_train, targets_train, sliding_window_num_blocks).backward()
            inputs_train, targets_train = train_loader.next_batch()
    if train_accumulation_steps != 1:
        for p in model.parameters():
            p.grad /= train_accumulation_steps
    # step the optimizer and scheduler
    optimizer.step()
    scheduler.step()
    # null the gradients
    model.zero_grad(set_to_none=True)
    # --------------- TRAINING SECTION END -------------------
    # everything that follows now is just diagnostics, prints, logging, etc.
    approx_time = training_time_ms + 1000 * (time.perf_counter() - t0)
    print0(
        f"step:{step+1}/{args.num_iterations} train_time:{approx_time:.0f}ms step_avg:{approx_time/timed_steps:.2f}ms"
    )

print0(
    f"peak memory consumption: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB"
)

# -------------------------------------------------------------------------
# clean up nice
dist.destroy_process_group()
