#include "correlation.h"
#include <chrono>
#include <cuda_runtime.h>
using namespace std;

void _checkCudaErrors(cudaError_t result, char const *const func, const char *const file, int const line) {
	if (result != cudaSuccess) {
		fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n", file, line,
				static_cast<unsigned int>(result), cudaGetErrorName(result), func);
		cudaDeviceReset();
		exit(EXIT_FAILURE);
	}
}
#define checkCudaErrors(val) _checkCudaErrors((val), #val, __FILE__, __LINE__)

std::vector<torch::Tensor> run_cuda_implementation(torch::Tensor input1, torch::Tensor input2, torch::Tensor grad_output, int max_displacement) {
	// nchw -> nhwc
	input1 = input1.permute({0, 2, 3, 1}).contiguous();
	input2 = input2.permute({0, 2, 3, 1}).contiguous();

	torch::Tensor output = correlation_forward_cuda(input1, input2, max_displacement);
	std::pair<torch::Tensor, torch::Tensor> grads_cuda = correlation_backward_cuda(grad_output, input1, input2, max_displacement);
	
	return {output, grads_cuda.first, grads_cuda.second};
}

std::vector<torch::Tensor> run_naive_implementation(torch::Tensor input1, torch::Tensor input2, torch::Tensor grad_output, int max_displacement) {
	int batch_size = input1.size(0), in_channels = input1.size(1), height = input1.size(2), width = input1.size(3);
	torch::Tensor padded_input2 = torch::nn::functional::detail::pad(input2, {max_displacement, max_displacement, max_displacement, max_displacement}, torch::kConstant, 0);

	std::vector<torch::Tensor> cost_volumes;
	for (int i = 0; i < 2 * max_displacement + 1; i++)
		for (int j = 0; j < 2 * max_displacement + 1; j++) {
			torch::Tensor cost = input1 * padded_input2.slice(2, i, i + height).slice(3, j, j + width);
			cost_volumes.push_back(torch::mean(cost, 1, true));
		}

	torch::Tensor output = torch::cat(cost_volumes, 1);
	output.backward(grad_output);

	return {output, input1.grad(), input2.grad()};
}

int main() {
	constexpr int batch_size = 32;
	constexpr int in_channels = 128;
	constexpr int height = 144;
	constexpr int width = 240;
	constexpr int max_displacement = 4;
	constexpr int out_channels = (max_displacement * 2 + 1) * (max_displacement * 2 + 1);

	if (!torch::cuda::is_available()) {
		cout << "CUDA is not available, exiting..." << endl;
		return 1;
	}

	torch::manual_seed(0);
	torch::Tensor input1 = torch::rand({batch_size, in_channels, height, width}, torch::requires_grad().device("cuda"));
	torch::Tensor input2 = torch::rand({batch_size, in_channels, height, width}, torch::requires_grad().device("cuda"));
	torch::Tensor grad_output = torch::rand({batch_size, out_channels, height, width}, torch::requires_grad(false).device("cuda"));

	cout << "Running NAIVE implementation of correlation... " << flush;
	auto naive_t1 = chrono::high_resolution_clock::now();
	std::vector<torch::Tensor> results_naive = run_naive_implementation(input1, input2, grad_output, max_displacement);
	checkCudaErrors(cudaDeviceSynchronize());
	auto naive_t2 = chrono::high_resolution_clock::now();
	cout << "(" << chrono::duration_cast<chrono::milliseconds>(naive_t2 - naive_t1).count() << "ms)" << endl;

	// warm up...
	for (int t = 0; t < 2; t++) {
		run_cuda_implementation(input1, input2, grad_output, max_displacement);
		checkCudaErrors(cudaDeviceSynchronize());
	}

	cout << "Running CUDA implementation of correlation... " << flush;
	auto cuda_t1 = chrono::high_resolution_clock::now();
	std::vector<torch::Tensor> results_cuda = run_cuda_implementation(input1, input2, grad_output, max_displacement);
	checkCudaErrors(cudaDeviceSynchronize());
	auto cuda_t2 = chrono::high_resolution_clock::now();
	cout << "(" << chrono::duration_cast<chrono::milliseconds>(cuda_t2 - cuda_t1).count() << "ms)" << endl;

	float diff = torch::mean(torch::abs(results_cuda[0].cpu() - results_naive[0].cpu())).data_ptr<float>()[0];
	cout << "Checking forward results... " << (diff < 1e-6 ? "OK" : "Failed") << endl;

	diff = torch::mean(torch::abs(results_cuda[1].cpu() - results_naive[1].cpu())).data_ptr<float>()[0];
	cout << "Checking backward results for input1... " << (diff < 1e-6 ? "OK" : "Failed") << endl;

	diff = torch::mean(torch::abs(results_cuda[2].cpu() - results_naive[2].cpu())).data_ptr<float>()[0];
	cout << "Checking backward results for input2... " << (diff < 1e-6 ? "OK" : "Failed") << endl;

	return 0;
}
