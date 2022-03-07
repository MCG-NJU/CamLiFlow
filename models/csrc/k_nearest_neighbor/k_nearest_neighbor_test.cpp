#include "k_nearest_neighbor.h"
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

torch::Tensor k_nearest_neighbor_std(torch::Tensor input_xyz, torch::Tensor query_xyz, int k) {
	int64_t batch_size = input_xyz.size(0), n_points_1 = input_xyz.size(1), n_points_2 = query_xyz.size(1);
	torch::Tensor dists = -2 * torch::matmul(query_xyz, input_xyz.permute({0, 2, 1}));
	dists += torch::sum(query_xyz.pow(2), -1).view({batch_size, n_points_2, 1});
	dists += torch::sum(input_xyz.pow(2), -1).view({batch_size, 1, n_points_1});
	return std::get<1>(dists.topk(k, 2, false));
}

int main() {
	constexpr int batch_size = 8;
	constexpr int n_points_input = 8192;
	constexpr int n_points_query = 8192;
	constexpr int k = 16;
	constexpr int dim = 3;

	if (!torch::cuda::is_available()) {
		cout << "CUDA is not available, exiting..." << endl;
		return 1;
	}

	torch::manual_seed(0);
	torch::Tensor input_xyz = torch::rand({batch_size, n_points_input, dim}).cuda();
	torch::Tensor query_xyz = torch::rand({batch_size, n_points_query, dim}).cuda();

	// warm up...
	for (int t = 0; t < 3; t++) {
		k_nearest_neighbor_cuda(input_xyz, query_xyz, k);
		k_nearest_neighbor_std(input_xyz, query_xyz, k);
	}
	checkCudaErrors(cudaDeviceSynchronize());

	cout << "Running KNN using custom CUDA implementation... " << flush;
	auto t1 = chrono::high_resolution_clock::now();
	torch::Tensor indices_gpu = k_nearest_neighbor_cuda(input_xyz, query_xyz, k);
	checkCudaErrors(cudaDeviceSynchronize());
	auto t2 = chrono::high_resolution_clock::now();
	cout << "(" << chrono::duration_cast<chrono::milliseconds>(t2 - t1).count() << "ms)" << endl;

	cout << "Running KNN using Torch's API... " << flush;
	t1 = chrono::high_resolution_clock::now();
	torch::Tensor indices_std = k_nearest_neighbor_std(input_xyz, query_xyz, k);
	checkCudaErrors(cudaDeviceSynchronize());
	t2 = chrono::high_resolution_clock::now();
	cout << "(" << chrono::duration_cast<chrono::milliseconds>(t2 - t1).count() << "ms)" << endl;

	torch::Scalar diff_num = (indices_std.cpu() != indices_gpu.cpu()).sum().item();
	torch::Scalar total_num = batch_size * n_points_query * k;
	cout << "Checking results... " << diff_num << " of " << total_num << " elements are mismatched." << endl;

	return 0;
}