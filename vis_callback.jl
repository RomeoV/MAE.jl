using Flux, FluxTraining
using FastVision: RGB
import ImageCore: colorview
struct VisualizationCallback <: FluxTraining.Callback 
  task
  device
end

cview(x_batch) = permutedims(x_batch, (3, 1, 2)) |> colorview(RGB)
function FluxTraining.on(
                ::FluxTraining.EpochEnd,
                ::FluxTraining.Phases.AbstractValidationPhase,
                cb::VisualizationCallback,
                learner)
  xs = first(learner.data[:validation])[1][:, :, :, 1:1] |> cb.device
  ys = learner.model(xs);
  # FastAI.showoutputbatch(ShowText(), cb.task, cpu.(xs), cpu.(ys))
  plt_lhs = cview(cpu(xs[:, :, :, 1]))
  plt_rhs = cview(cpu(ys[:, :, :, 1]))
  FastAI.showsample(ShowText(), cb.task, (plt_lhs, plt_rhs))
end
FluxTraining.stateaccess(::VisualizationCallback) = (data=FluxTraining.Read(), 
                                                     model=FluxTraining.Read(), )
FluxTraining.runafter(::VisualizationCallback) = (Metrics,)
