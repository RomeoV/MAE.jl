import Metalhead: PatchEmbedding, ClassTokens, ViPosEmbedding
import Metalhead: MultiHeadSelfAttention, prenorm
import Random: randperm
import NNlib: gather!
import Zygote.ChainRules: @ignore_derivatives
using SimpleConfig
using Flux

struct MAEEncoder
  patch_emb
  class_token
  pos_emb
  norm
  backbone
  npatches
end
Flux.@functor MAEEncoder

function MAEEncoder(img_size::Tuple{I, I}; embedplanes::I=128, patch_size::Tuple{I, I}=(16, 16), nblocks::I=3) where I <: Integer
  @assert all(==(0), img_size .% patch_size) "`img_size` must be cleanly divisible by `patch_size`."
  npatches = img_size .÷ patch_size |> prod
  patch_emb = PatchEmbedding(img_size; embedplanes, patch_size)
  class_token = ClassTokens(embedplanes)
  pos_emb =   ViPosEmbedding(embedplanes, npatches+1)
  emb_norm =  LayerNorm(embedplanes)
  blks_enc =  [prenorm(embedplanes, MultiHeadSelfAttention(embedplanes)) for _ in 1:nblocks];
  model_enc = Chain(blks_enc...);
  return MAEEncoder( patch_emb, class_token, pos_emb, emb_norm, model_enc, npatches )
end

function (m::MAEEncoder)(x)
  BATCH_SIZE = size(x)[end]
  # 1)
  x = m.patch_emb(x);

  # 2)
  x = @ignore_derivatives x .+ m.pos_emb.vectors[:, 2:end]

  # 3)
  perm = @ignore_derivatives randperm(m.npatches);
  idx_keep = perm[1:m.npatches÷4]
  dst = @ignore_derivatives 0*similar(x, size(x, 1), length(idx_keep), BATCH_SIZE);  # preallocate this!
  for (x_, dst_) in zip(eachslice(x;   dims=3),
                        eachslice(dst; dims=3))
    gather!(dst_, x_, idx_keep);
  end
  x_masked = dst;

  # 4)
  class_emb = @ignore_derivatives repeat(m.class_token.token .+ m.pos_emb.vectors[:, 1],
                             1, 1, BATCH_SIZE);
  x = cat(class_emb, x_masked;
          dims=2);

  # 5)
  y = m.backbone(x);
  y = m.norm(y);
  return y, perm
end

struct MAEDecoder
  decoder_emb
  pos_emb
  norm
  mask_tokens
  backbone
  decoder_pred
  npatches
end
Flux.@functor MAEDecoder

function MAEDecoder(in_dim::I, npatches; embedplanes::I=64, patch_size::Tuple{I, I}=(16, 16), nblocks::I=3) where I <: Integer
  decoder_emb = Dense(in_dim, embedplanes)
  pos_emb = ViPosEmbedding(embedplanes, npatches+1)
  norm = LayerNorm(embedplanes)
  mask_tokens = ones(embedplanes, 1, 1)
  blks_dec = [prenorm(embedplanes, MultiHeadSelfAttention(embedplanes)) for _ in 1:nblocks];
  model_dec = Chain(blks_dec...);
  decoder_pred = Dense(embedplanes, prod(patch_size)*3)
  return MAEDecoder( decoder_emb, pos_emb, norm, mask_tokens, model_dec, decoder_pred, npatches )
end

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

  # 3)
  dst = @ignore_derivatives 0*similar(x, DIM_SIZE, m.npatches, BATCH_SIZE)
  for (x_, dst_) in zip(eachslice(x;   dims=3),
                        eachslice(dst; dims=3))
    gather!(dst_, x_, perm_rev);
  end
  x = dst;
  x = cat(cls, x; dims=2);

  # 4)
  x = @ignore_derivatives x .+ m.pos_emb.vectors;

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

imgs = rand(Float32, 224, 224, 3, 7);
m = make_model()
m(imgs);
gs = gradient(Flux.params(m)) do
  m(imgs) |> sum  # check the gradients...
end
