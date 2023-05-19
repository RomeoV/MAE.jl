using Revise
includet("own_mae.jl")
imgs = rand(Float32, 224, 224, 3, 7);
results = []
grads = []
for bs in [nothing, 7]
  seed!(1)
  m = make_model();
  m(copy(imgs));
  s, gs = withgradient(Flux.params(m)) do
    m(imgs) |> sum  # check the gradients...
  end;
  push!(results, s)
  push!(grads, gs)
end
@assert results[1] ≈ results[2]
@assert length(grads[1].params) == length(grads[2].params)
lhs = sum(sum(v) for v in values(grads[1].grads) if v isa Array);
rhs = sum(sum(v) for v in values(grads[2].grads) if v isa Array);
# @assert (lhs ≈ rhs) "The gradients are not equal! See $lhs != $rhs"

@info "Starting benchmarks"
benchmarks = Dict(
    (bs, dev)=>(@benchmark [ gradient(ps) do
                               m(imgs_dev) |> sum
                             end
                             for _ in 1:20 ] setup=begin
                  seed!(1) 
                  m = make_model() |> $dev
                  bs_ = (!isnothing($bs) ? $bs : 16)
                  imgs_dev = rand(Float32, 224, 224, 3, bs_) |> $dev 
                  ps = Flux.params(m) 
                end
    )
    for bs  in [nothing, 16, 32],
        dev in [gpu]
        # dev in [cpu, gpu]
)  # conclusion: preallocating is about 10-20% faster

for bs in [nothing, 7]
  imgs = rand(Float32, 224, 224, 3, 7)
  m = make_model();
  m(imgs);
  s, gs = withgradient(Flux.params(m)) do
    m(imgs) |> sum  # check the gradients...
  end;
  # check the gradients...
  failed_grads = [k[1:3] for (k, v) in gs.grads if isnothing(v)];
  @assert isempty(failed_grads)
end

# begin
# m_gpu = m |> gpu;
# imgs_gpu = imgs |> gpu;
# m_gpu(imgs_gpu);
# for _ in ProgressBar(1:100)
#   m_gpu(imgs_gpu);
# end

