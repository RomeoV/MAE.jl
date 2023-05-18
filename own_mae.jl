import Metalhead: PatchEmbedding, ClassTokens, ViPosEmbedding
import Metalhead: MultiHeadSelfAttention,
                  prenorm  # applies LayerNorm, then the next module
import Random: randperm
import NNlib: gather!  # we use this for shuffling and unshuffling
import Zygote.ChainRules: @ignore_derivatives
using SimpleConfig
using Flux
import MLUtils: zeros_like, ones_like
import Random: seed!
seed!(1)

struct MAEEncoder
  patch_emb
  class_token
  pos_emb
  norm
  backbone
  npatches
end
Flux.@functor MAEEncoder
Flux.trainable(m::MAEEncoder) = (m.patch_emb, m.class_token, m.norm, m.backbone)  # no pos embedding

function MAEEncoder(img_size::Tuple{I, I};
                    embedplanes::I=128,
                    patch_size::Tuple{I,I}=(16,16),
                    nblocks::I=3) where I <: Integer
  @assert all(==(0), img_size .% patch_size) "`img_size` must be cleanly divisible by `patch_size`."
  npatches = img_size .÷ patch_size |> prod
  patch_emb = PatchEmbedding(img_size; embedplanes, patch_size)
  class_token = ClassTokens(embedplanes)
  pos_emb =   ViPosEmbedding(embedplanes, npatches+1)
  emb_norm =  LayerNorm(embedplanes)
  model_enc = Chain([prenorm(embedplanes,
                             MultiHeadSelfAttention(embedplanes))
                     for _ in 1:nblocks]...);
  return MAEEncoder( patch_emb, class_token, pos_emb, emb_norm, model_enc, npatches )
end

function (m::MAEEncoder)(x)
  BATCH_SIZE = size(x)[end]
  # 1)
  x = m.patch_emb(x);

  # 2)
  x = x .+ m.pos_emb.vectors[:, 2:end]

  # 3)
  perm = @ignore_derivatives randperm(m.npatches)
  idx_keep = perm[1:m.npatches÷4]
  # idx_keep = begin
  #   idx_keep = perm[1:m.npatches÷4]
  #   [CartesianIndex(idx, b) for idx in idx_keep,
  #                               b in 1:BATCH_SIZE]
  # end
  # dst =  ones_like(x, (size(x, 1), size(idx_keep, 1), BATCH_SIZE));  # preallocate this!
  # gather!(dst, x, idx_keep)
  # x_masked = dst;
  x_masked = cat([gather(x_, idx_keep) for x_ in eachslice(x; dims=3)]...;
                 dims=3)

  # 4)
  class_emb = repeat(m.class_token.token .+ m.pos_emb.vectors[:, 1],
                     1, 1, BATCH_SIZE);
  x = cat(class_emb, x_masked;
          dims=2);

  # 5)
  y = m.backbone(x);
  y = m.norm(y);
  return y, perm
end

const AA3{T} = AbstractArray{T, 3}
struct MAEDecoder{M<:AA3}
  decoder_emb
  pos_emb
  norm
  mask_tokens::M
  backbone
  decoder_pred
  npatches::Integer
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
                    nblocks::I=3) where I <: Integer
  decoder_emb = Dense(in_dim, embedplanes)
  pos_emb = ViPosEmbedding(embedplanes, npatches+1)
  norm = LayerNorm(embedplanes)
  mask_tokens = 1//2*ones(Float32, embedplanes, 1, 1)  # currently this receives no gradient!!
  model_dec = Chain([prenorm(embedplanes,
                             MultiHeadSelfAttention(embedplanes))
                     for _ in 1:nblocks]...);
  decoder_pred = Dense(embedplanes, prod(patch_size)*3)
  return MAEDecoder( decoder_emb, pos_emb, norm, mask_tokens, model_dec, decoder_pred, npatches )
end

dst = zeros(Float32, 64, 14*14, 7)
function (m::MAEDecoder)((x, perm)::Tuple)
  BATCH_SIZE = size(x)[end]
  perm_rev = @ignore_derivatives sortperm(perm)

  # 1)
  x = m.decoder_emb(x);
  DIM_SIZE = size(x, 1)

  # 2)
  cls, x = x[:, 1:1, :], x[:, 2:end, :];
  x = cat(  x
          , repeat(m.mask_tokens,
                   1, m.npatches - size(x, 2), BATCH_SIZE)
          ; dims=2);

  # # 3)
  # currently, the gradient doesn't flow right through this...
  # dst = zeros_like(x, (DIM_SIZE, m.npatches, BATCH_SIZE))
  # dst = zeros(Float32, DIM_SIZE, m.npatches, BATCH_SIZE)
  # gather!(dst, x, [CartesianIndex(idx, b) for idx in perm_rev,
  #                                             b in 1:BATCH_SIZE])
  # x = 0*x + dst;
  x = cat([gather(x_, perm_rev) for x_ in eachslice(x, dims=3)]...;
          dims=3)
  # x_ = gather(x, [CartesianIndex(idx, b) for idx in perm_rev,
  #                                             b in 1:BATCH_SIZE])
  @assert size(x, 2) == m.npatches
  x = cat(cls, x; dims=2);

  # 4)
  x = x .+ m.pos_emb.vectors;

  # 5)
  x = m.backbone(x);
  x = m.norm(x);
  x = m.decoder_pred(x)
  x = x[:, 2:end, :];
  x
end

make_model() = Chain(MAEEncoder((224, 224)), MAEDecoder(128, 14*14))


## ENCODER
# 1) patch embed
# 2) add pos embed
# 3) mask x
# 4) make class token + pos embed, and prepend
# 5) send through transformer and normalize

## Decoder
# 1) redo embedding
# 2) remove class token, append learnable mask token
# 3) unshuffle
# 4) add pos embedding
# 5) send through transformer and normalize

imgs = rand(Float32, 224, 224, 3, 32);
m = make_model()
m(imgs);
gs = gradient(Flux.params(m)) do
  m(imgs) |> sum  # check the gradients...
end;
# check the gradients...
[k[1:3] for (k, v) in gs.grads if isnothing(v)]

m_gpu = m |> gpu;
imgs_gpu = imgs |> gpu;
m_gpu(imgs_gpu);
for _ in 1:100
  m_gpu(imgs_gpu);
end

