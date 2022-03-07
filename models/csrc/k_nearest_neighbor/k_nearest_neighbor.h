#pragma once

#include <torch/extension.h>

torch::Tensor k_nearest_neighbor_cuda(torch::Tensor input_xyz, torch::Tensor query_xyz, int k);