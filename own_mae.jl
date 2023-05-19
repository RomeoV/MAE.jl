import Metalhead: PatchEmbedding, ClassTokens, ViPosEmbedding
import Metalhead: MultiHeadSelfAttention,
                  prenorm  # applies LayerNorm, then the next module
import Random: randperm
import NNlib: gather, gather!  # we use this for shuffling and unshuffling
import Zygote: withgradient, Buffer
import Zygote.ChainRules: @ignore_derivatives
using SimpleConfig
using Flux
import MLUtils: zeros_like, ones_like
using ProgressBars
import Random: seed!
using BenchmarkTools
seed!(1)

const AA3{T} = AbstractArray{T, 3}

struct MAEEncoder{P}
  patch_emb
  class_token
  pos_emb
  norm
  backbone
  npatches::Integer
  npatches_keep::Integer
  dst_prealloc::P
end
Flux.@functor MAEEncoder
Flux.trainable(m::MAEEncoder) = (m.patch_emb, m.class_token, m.norm, m.backbone)  # no pos embedding

function MAEEncoder(img_size::Tuple{I, I};
                    embedplanes::I=128,
                    patch_size::Tuple{I,I}=(16,16),
                    nblocks::I=3,
                    pct_patches_keep=.25,
                    prealloc_batch_size::Union{I, Nothing}=nothing) where I <: Integer
  @assert all(==(0), img_size .% patch_size) "`img_size` must be cleanly divisible by `patch_size`."
  npatches = img_size .÷ patch_size |> prod
  npatches_keep = floor(Int, npatches*pct_patches_keep)
  patch_emb = PatchEmbedding(img_size; embedplanes, patch_size)
  class_token = ClassTokens(embedplanes)
  pos_emb =   ViPosEmbedding(embedplanes, npatches+1)
  emb_norm =  LayerNorm(embedplanes)
  model_enc = Chain([prenorm(embedplanes,
                             MultiHeadSelfAttention(embedplanes))
                     for _ in 1:nblocks]...)
  dst_prealloc = (!isnothing(prealloc_batch_size) ? zeros(Float32, embedplanes, npatches_keep, prealloc_batch_size)
                                                  : nothing)
  return MAEEncoder( patch_emb, class_token, pos_emb, emb_norm, model_enc, npatches, npatches_keep, dst_prealloc )
end

function shuffle_patches(dst::Nothing, x, idx_keep, batch_size)
  x_shuffled = cat([gather(x_, idx_keep) for x_ in eachslice(x; dims=3)]...;
                   dims=3)
  return x_shuffled
end

# function shuffle_patches(dst::AA3, x, idx_keep, batch_size)
#   gather!(dst, x, [CartesianIndex(idx, b) for idx in idx_keep,
#                                               b in 1:batch_size])
#   x_shuffled = 0*x[:, 1:length(idx_keep), :] + dst;  # just setting x = dst doesn't carry the gradient properly...
#   return x_shuffled
# end

# function shuffle_patches(dst::AA3, x, idx_keep, batch_size)
#   map((dst_, x_)::Tuple->gather!(dst_, x_, idx_keep),
#       zip(eachslice(dst; dims=3),
#           eachslice(x;   dims=3)))
#   # for (dst_, x_) in zip(eachslice(dst; dims=3),
#   #                       eachslice(x;   dims=3))
#   #   gather!(dst_, x_, idx_keep)
#   # end
#   x_shuffled = copy(dst)
#   return x_shuffled
# end

function shuffle_patches(dst::AA3, x, idx_keep, batch_size)
  x_shuffled = x[:, idx_keep, :]
  return x_shuffled
end

function shuffle_patches(dst::Buffer, x, idx_keep, batch_size)
  @info "Using buffer :)"
  gather!(dst, x, [CartesianIndex(idx, b) for idx in idx_keep,
                                              b in 1:batch_size])
  x_shuffled = copy(dst);  # just setting x = dst doesn't carry the gradient properly...
  return x_shuffled
end

## Encoder
# 1) encode each patch into a 1d embedding
# 2) add pos embed to patches
# 3) shuffle patches and keep a small percentage
# 4) make and prepend class token w/ positional embedding
# 5) apply transformer
function (m::MAEEncoder)(x)
  BATCH_SIZE = size(x)[end]
  # 1) encode each patch into a 1d embedding
  # (usually conv|>flatten or flatten|>dense)
  x = m.patch_emb(x);

  # 2) add pos embed to patches
  x = x .+ m.pos_emb.vectors[:, 2:end]

  # 3) shuffle patches and keep a small percentage
  # This dispatches, depending on whether we have preallocated storage.
  perm = @ignore_derivatives randperm(m.npatches)
  idx_keep = perm[1:m.npatches_keep]
  x_masked = x[:, idx_keep, :]

  # 4) make and prepend class token w/ positional embedding
  class_emb = repeat(m.class_token.token .+ m.pos_emb.vectors[:, 1],
                     1, 1, BATCH_SIZE);
  x = cat(class_emb, x_masked;
          dims=2);

  # 5) apply transformer
  x = m.backbone(x);
  x = m.norm(x);
  return x, perm
end

struct MAEDecoder{M<:AA3, P}
  decoder_emb
  pos_emb
  norm
  mask_tokens::M
  backbone
  decoder_pred
  npatches::Integer
  npatches_keep::Integer
  dst_prealloc::P
end
Flux.@functor MAEDecoder
Flux.trainable(m::MAEDecoder) = (m.decoder_emb,
                                 m.norm,
                                 m.mask_tokens,
                                 m.backbone,
                                 m.decoder_pred)  # no pos embedding

function MAEDecoder(in_dim::I, npatches;
                    embedplanes::I=64,
                    patch_size::Tuple{I,I}=(16,16),
                    nblocks::I=3,
                    pct_patches_keep=0.25,
                    prealloc_batch_size::Union{I, Nothing}=0) where I <: Integer
  npatches_keep = floor(Int, npatches*pct_patches_keep)
  decoder_emb = Dense(in_dim, embedplanes)
  pos_emb = ViPosEmbedding(embedplanes, npatches+1)
  norm = LayerNorm(embedplanes)
  mask_tokens = 1//2*ones(Float32, embedplanes, 1, 1)  # currently this receives no gradient!!
  model_dec = Chain([prenorm(embedplanes,
                             MultiHeadSelfAttention(embedplanes))
                     for _ in 1:nblocks]...);
  decoder_pred = Dense(embedplanes, prod(patch_size)*3)
  dst_prealloc = (!isnothing(prealloc_batch_size) ? 1//3*ones(Float32, embedplanes, npatches, prealloc_batch_size)
                                                  : nothing)
  return MAEDecoder( decoder_emb, pos_emb, norm, mask_tokens, model_dec, decoder_pred, npatches, npatches_keep, dst_prealloc )
end


## Decoder
# 1) redo embedding
# 2) remove class token, append learnable mask token
# 3) unshuffle
# 4) add pos embedding
# 5) send through transformer and normalize
function (m::MAEDecoder)((x, perm)::Tuple)
  BATCH_SIZE = size(x)[end]

  # 1) project embedding into new space
  x = m.decoder_emb(x);
  DIM_SIZE = size(x, 1)

  # 2) remove class token, append learnable mask token
  cls, x = x[:, 1:1, :], x[:, 2:end, :];
  x = cat(  x
          , repeat(m.mask_tokens,
                   1, m.npatches - size(x, 2), BATCH_SIZE)
          ; dims=2);

  # 3) unshuffle and prepend class token again
  # x = unshuffle_patches(m.dst_prealloc, x, perm, BATCH_SIZE)
  x = x[:, sortperm(perm), :]
  @assert size(x, 2) == m.npatches
  x = cat(cls, x; dims=2);

  # 4) add new positional embedding
  x = x .+ m.pos_emb.vectors;

  # 5) apply transformer
  x = m.backbone(x);
  x = m.norm(x);
  x = m.decoder_pred(x)
  x = x[:, 2:end, :];
  x
end

make_model(; batch_size=nothing) = Chain(MAEEncoder((224, 224); prealloc_batch_size=batch_size),
                                         MAEDecoder(128, 14*14; prealloc_batch_size=batch_size))
