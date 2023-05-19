includet("own_mae.jl")
includet("vis_callback.jl")
using FastAI, FastVision
import Flux
import MLUtils: _default_executor
import MLUtils.Transducers: ThreadedEx
import CUDA
CUDA.allowscalar(false)
# ThreadPoolEx gave me problems, see https://github.com/JuliaML/MLUtils.jl/issues/142
_default_executor() = ThreadedEx()

(data_img, data_lab), blocks = load(datarecipes()["cifar10"])
data = mapobs(s->let s = loadfile(s)
                (s, s)
              end,
              data_img.data)

task = SupervisedTask(
    (Image{2}(), Image{2}()),
    (
        ProjectiveTransforms((32, 32)),
        ImagePreprocessing(),
    )
)

traindl, validdl = taskdataloaders(data, task, 16);
model = Chain(MAEEncoder((32, 32); patch_size=(4, 4)),
              MAEDecoder((32, 32), 128; patch_size=(4, 4)));
optimizer = Flux.AdamW(3e-4);
learner = Learner(model, Flux.Losses.mse;
                  data = (traindl, validdl),
                  optimizer, callbacks=[ProgressPrinter(),
                                        ToGPU(),
                                        VisualizationCallback(task, gpu)])
fit!(learner, 3)
