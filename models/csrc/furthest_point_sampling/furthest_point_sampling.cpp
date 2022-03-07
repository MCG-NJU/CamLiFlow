#include "furthest_point_sampling.h"

void furthest_point_sampling_kernel_wrapper(float* batched_points_xyz, float* batched_dists_temp, int n_batch, int n_points, int n_samples, int64_t* batched_furthest_indices);

torch::Tensor furthest_point_sampling_cuda(torch::Tensor points_xyz, const int n_samples) {
	TORCH_CHECK(points_xyz.is_contiguous(), "points_xyz must be a contiguous tensor");
	TORCH_CHECK(points_xyz.is_cuda(), "points_xyz must be a CUDA tensor");
	TORCH_CHECK(points_xyz.scalar_type() == torch::ScalarType::Float, "points_xyz must be a float tensor");

	int64_t batch_size = points_xyz.size(0), n_points = points_xyz.size(1);
	torch::Tensor furthest_indices = torch::empty({ batch_size, n_samples }, torch::TensorOptions().dtype(torch::kInt64).device(points_xyz.device()));
	torch::Tensor dists_temp = torch::ones({batch_size, n_points}, torch::TensorOptions().dtype(torch::kFloat32).device(points_xyz.device())) * 1e10;
	furthest_point_sampling_kernel_wrapper(points_xyz.data_ptr<float>(), dists_temp.data_ptr<float>(), batch_size, n_points, n_samples, furthest_indices.data_ptr<int64_t>());

	return furthest_indices;
}

#ifdef TORCH_EXTENSION_NAME
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
	m.def("_furthest_point_sampling_cuda", &furthest_point_sampling_cuda, "CUDA implementation of furthest-point-sampling (FPS)");
}
#endif