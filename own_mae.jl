import Metalhead: PatchEmbedding, ClassTokens, ViPosEmbedding,
                  MultiHeadSelfAttention, transformer_encoder,
                  prenorm  # applies LayerNorm, then the next module
import Random: randperm
import Zygote.ChainRules: @ignore_derivatives
import Flux
import Flux: LayerNorm, Dense, Chain
using Tullio
using KernelAbstractions, CUDA
using CUDA.CUDAKernels
# import Tullio: @tullio
CUDA.allowscalar(false)


const AA3{T} = AbstractArray{T, 3}

struct MAEEncoder
  patch_emb
  class_token
  pos_emb
  norm
  backbone
  img_size::Tuple{Integer, Integer}
  patch_size::Tuple{Integer, Integer}
  npatches::Integer
  npatches_keep::Integer
end
Flux.@functor MAEEncoder
Flux.trainable(m::MAEEncoder) = (m.patch_emb, m.class_token, m.norm, m.backbone)  # no pos embedding

function MAEEncoder(img_size::Tuple{I, I};
                    embedplanes::I=128,
                    patch_size::Tuple{I,I}=(16,16),
                    nblocks::I=12,
                    pct_patches_keep=.25) where I <: Integer
  @assert all(==(0), img_size .% patch_size) "`img_size` must be cleanly divisible by `patch_size`."
  npatches = img_size .÷ patch_size |> prod
  npatches_keep = floor(Int, npatches*pct_patches_keep)
  patch_emb = PatchEmbedding(img_size; embedplanes, patch_size)
  class_token = ClassTokens(embedplanes)
  pos_emb =   ViPosEmbedding(embedplanes, npatches+1)
  emb_norm =  LayerNorm(embedplanes)
  # model_enc = Chain([prenorm(embedplanes,
  #                            MultiHeadSelfAttention(embedplanes))
  #                    for _ in 1:nblocks]...)
  model_enc = transformer_encoder(embedplanes, nblocks, 8)
  return MAEEncoder( patch_emb, class_token, pos_emb, emb_norm, model_enc, img_size, patch_size, npatches, npatches_keep )
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
  # x_masked = shuffle_patches(m.dst_prealloc, x, idx_keep, BATCH_SIZE)
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

struct MAEDecoder{M<:AA3}
  decoder_emb
  pos_emb
  norm
  mask_tokens::M
  backbone
  decoder_pred
  img_size::Tuple{Integer, Integer}
  patch_size::Tuple{Integer, Integer}
  npatches::Integer
  npatches_keep::Integer
end
Flux.@functor MAEDecoder
Flux.trainable(m::MAEDecoder) = (m.decoder_emb,
                                 m.norm,
                                 m.mask_tokens,
                                 m.backbone,
                                 m.decoder_pred)  # no pos embedding

function MAEDecoder(img_size::Tuple{I, I}, input_emb_dim::I;
                    embedplanes::I=64,
                    patch_size::Tuple{I,I}=(16,16),
                    nblocks::I=3,
                    pct_patches_keep=0.25) where I <: Integer
  @assert all(==(0), img_size .% patch_size) "`img_size` must be cleanly divisible by `patch_size`."
  npatches = img_size .÷ patch_size |> prod
  npatches_keep = floor(Int, npatches*pct_patches_keep)
  decoder_emb = Dense(input_emb_dim, embedplanes)
  pos_emb = ViPosEmbedding(embedplanes, npatches+1)
  norm = LayerNorm(embedplanes)
  mask_tokens = 1//2*ones(Float32, embedplanes, 1, 1)  # currently this receives no gradient!!
  model_dec = transformer_encoder(embedplanes, nblocks, 8)
  decoder_pred = Dense(embedplanes, prod(patch_size)*3)
  return MAEDecoder( decoder_emb, pos_emb, norm, mask_tokens, model_dec, decoder_pred, img_size, patch_size, npatches, npatches_keep )
end


## Decoder
# 1) redo embedding
# 2) remove class token, append learnable mask token
# 3) unshuffle
# 4) add pos embedding
# 5) send through transformer and normalize
function (m::MAEDecoder)(x::AA3, perm::Vector)
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

  h, w = m.img_size .÷ m.patch_size
  x = reshape(x, 3, m.patch_size..., h, w, BATCH_SIZE)
  # @tullio x_[p1, h, p2, w, c, b] := x[c, p1, p2, h, w, b]
  x = permutedims(x, (2, 4, 3, 5, 1, 6))
  x = reshape(x, h*m.patch_size[1], w*m.patch_size[2], 3, BATCH_SIZE)
  x
end
(m::MAEDecoder)((x, perm)::Tuple) = m(x, perm)

make_model(; batch_size=nothing) = Chain(MAEEncoder((224, 224)),
                                         MAEDecoder((224, 224), 128))
