#pragma once

#include <torch/extension.h>

torch::Tensor correlation_forward_cuda(torch::Tensor& input1, torch::Tensor& input2, int max_displacement);
std::pair<torch::Tensor, torch::Tensor> correlation_backward_cuda(torch::Tensor& grad_output, torch::Tensor& input1, torch::Tensor& input2, int max_displacement);
