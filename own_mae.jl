import Metalhead: PatchEmbedding, ClassTokens, ViPosEmbedding
import Metalhead: MultiHeadSelfAttention
import Random: randperm
import NNlib: scatter, gather
import Zygote: @ignore

## Encoder parts
# params = imgsize::Tuple, planes::Integer, patch_size::Tuple, nblocks::Integer
patch_emb = PatchEmbedding((224, 224); embedplanes=128, patch_size=(16, 16))
class_tok = ClassTokens(128)
vi_pos_emb = ViPosEmbedding(128, 14*14+1)
emb_norm = LayerNorm(128)
blks_enc = [MultiHeadSelfAttention(128) for _ in 1:3];
model_enc = Chain(blks_enc...);

## Decoder parts
# params = planes_in::Integer, planes::Integer, patch_size::Tuple, nblocks::Integer
decoder_encode = Dense(128, 64)
vi_pos_emb2 = ViPosEmbedding(64, 14*14+1)
dec_norm = LayerNorm(64)
mask_tokens = ones(64, 1, 1)
blks_dec = [MultiHeadSelfAttention(64) for _ in 1:3];
model_dec = Chain(blks_dec...);

# emb = patch_emb(imgs);
# emb_ = class_tok(emb);
# emb__ = vi_pos_emb(emb_);
# size(emb), size(emb_), size(emb__)

ps = Flux.params([class_tok, Params(mask_tokens), model_enc, model_dec, decoder_encode])

# we dont' need the class token for mae
# now we select a number of patches = tokens at random


## ENCODER
# 1) patch embed
# 2) add pos embed
# 3) mask x
# 4) make class token + pos embed, and prepend
# 5) send through transformer and normalize

function run_model(imgs)
  # 1)
  x = patch_emb(imgs);

  # 2)
  x = x .+ vi_pos_emb.vectors[:, 2:end]

  # 3)
  perm = @ignore randperm(14*14);
  perm_rev = @ignore sortperm(perm);
  # x_masked = gather(x, idx_keep_);
  dst = zeros(Float32, 128, ((14*14)÷4), 7);
  for (x_, dst_) in zip(eachslice(x;   dims=3),
                        eachslice(dst; dims=3))
    gather!(dst_, x_, idx_keep);
  end
  x_masked = dst;

  # 4)
  class_emb = repeat(class_tok.token .+ vi_pos_emb.vectors[:, 1],
                     1, 1, 7);
  x = cat(class_emb, x_masked; dims=2);

  # 5)
  y = model_enc(x);
  y = emb_norm(y);

  ## Decoder
  # 1) redo embedding
  # 2) remove class token, append learnable mask token
  # 3) unshuffle
  # 4) add pos embedding
  # 5) send through transformer and normalize

  # 1)
  x = decoder_encode(y);

  # 2)
  cls, x = x[:, 1:1, :], x[:, 2:end, :];
  x = cat(  x
          , repeat(mask_tokens,
                 1, 14*14 - ((14*14)÷4), 7)
          ; dims=2);

  # 3)
  dst = zeros(64, 14*14, 7)
  for (x_, dst_) in zip(eachslice(x;   dims=3),
                        eachslice(dst; dims=3))
    gather!(dst_, x_, perm_rev);
  end
  x = dst;
  x = cat(cls, x; dims=2);

  # 4)
  x = x.+ vi_pos_emb2.vectors;

  # 5)
  y = model_dec(x);
  y = dec_norm(y);
  y = y[:, 2:end, :];
  y
end

imgs = rand(Float32, 224, 224, 3, 7);
gs = gradient(ps) do
  run_model(imgs) |> sum
end
