#include "furthest_point_sampling.h"
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

torch::Tensor furthest_point_sampling_std(torch::Tensor points_xyz, const int n_samples) {
	int64_t batch_size = points_xyz.size(0), n_points = points_xyz.size(1);
	torch::Tensor furthest_indices = torch::empty({batch_size, n_samples}, torch::TensorOptions().dtype(torch::kInt64));

	for (int64_t b = 0; b < batch_size; b++) {
		torch::Tensor dists = torch::ones(n_points) * 1e10;
		int curr_furthest_idx = 0;
		for (int s = 0; s < n_samples; s++) {
			furthest_indices[b][s] = curr_furthest_idx;
			dists = torch::min(dists, torch::sum((points_xyz[b] - points_xyz[b][curr_furthest_idx]).pow(2), -1));
			curr_furthest_idx = dists.argmax().data_ptr<int64_t>()[0];
		}
	}

	return furthest_indices;
}

int main() {
	constexpr int batch_size = 64;
	constexpr int n_points = 4096;
	constexpr int n_samples = 1024;

	if (!torch::cuda::is_available()) {
		cout << "CUDA is not available, exiting..." << endl;
		return 1;
	}

	torch::manual_seed(0);
	torch::Tensor points_xyz = torch::rand({batch_size, n_points, 3}), points_xyz_cuda = points_xyz.cuda();

	// warm up...
	for (int t = 0; t < 3; t++) furthest_point_sampling_cuda(points_xyz_cuda, n_samples);
	checkCudaErrors(cudaDeviceSynchronize());

	cout << "Running furthest-point-sampling on GPU... " << flush;
	auto t1 = chrono::high_resolution_clock::now();
	torch::Tensor indices_gpu = furthest_point_sampling_cuda(points_xyz_cuda, n_samples);
	checkCudaErrors(cudaDeviceSynchronize());
	auto t2 = chrono::high_resolution_clock::now();
	cout << "(" << chrono::duration_cast<chrono::milliseconds>(t2 - t1).count() << "ms)" << endl;

	cout << "Running furthest-point-sampling on CPU... " << flush;
	t1 = chrono::high_resolution_clock::now();
	torch::Tensor indices_std = furthest_point_sampling_std(points_xyz, n_samples);
	t2 = chrono::high_resolution_clock::now();
	cout << "(" << chrono::duration_cast<chrono::milliseconds>(t2 - t1).count() << "ms)" << endl;

	cout << "Checking results... " << (torch::equal(indices_std.cpu(), indices_gpu.cpu()) ? "OK" : "Failed") << endl;
}