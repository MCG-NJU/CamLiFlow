#pragma once

#include <torch/extension.h>

torch::Tensor furthest_point_sampling_cuda(torch::Tensor points_xyz, const int n_samples);
