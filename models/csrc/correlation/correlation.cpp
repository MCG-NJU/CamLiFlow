#include "correlation.h"

void correlation_forward_kernel_wrapper(float* output, const float* input1, const float* input2,
										int n_batches, int in_channels, int height, int width, int max_displacement);

void correlation_backward_kernel_wrapper(const float* grad_output,
										 float* grad_input1, float* grad_input2,
										 const float* input1, const float* input2,
										 int n_batches, int in_channels, int height, int width, int max_displacement);

torch::Tensor correlation_forward_cuda(torch::Tensor& input1, torch::Tensor& input2, int max_displacement) {
	TORCH_CHECK(input1.is_contiguous(), "input1 must be a contiguous tensor");
	TORCH_CHECK(input2.is_contiguous(), "input2 must be a contiguous tensor");

	int batch_size = input1.size(0), height = input1.size(1), width = input1.size(2), in_channels = input1.size(3);
	int out_channels = (max_displacement * 2 + 1) * (max_displacement * 2 + 1);
	torch::Tensor output = torch::zeros({batch_size, out_channels, height, width}, torch::device(input1.device()));

	correlation_forward_kernel_wrapper(output.data_ptr<float>(), input1.data_ptr<float>(), input2.data_ptr<float>(),
									   batch_size, in_channels, height, width, max_displacement);
	return output;
}

std::pair<torch::Tensor, torch::Tensor> correlation_backward_cuda(torch::Tensor& grad_output, torch::Tensor& input1, torch::Tensor& input2, int max_displacement) {
	int batch_size = input1.size(0), height = input1.size(1), width = input1.size(2), in_channels = input1.size(3);
	torch::Tensor grad_input1 = torch::empty({batch_size, in_channels, height, width}, torch::device(input1.device()));
	torch::Tensor grad_input2 = torch::empty({batch_size, in_channels, height, width}, torch::device(input2.device()));

	correlation_backward_kernel_wrapper(grad_output.data_ptr<float>(),
										grad_input1.data_ptr<float>(), grad_input2.data_ptr<float>(),
										input1.data_ptr<float>(), input2.data_ptr<float>(),
										batch_size, in_channels, height, width, max_displacement);

	return std::pair<torch::Tensor, torch::Tensor>(grad_input1, grad_input2);
}

#ifdef TORCH_EXTENSION_NAME
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
	m.def("_correlation_forward_cuda", &correlation_forward_cuda, "Correlation forward pass (CUDA)");
	m.def("_correlation_backward_cuda", &correlation_backward_cuda, "Correlation backward pass (CUDA)");
}
#endif
