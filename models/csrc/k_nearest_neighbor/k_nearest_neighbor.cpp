#include "k_nearest_neighbor.h"

void k_nearest_neighbor_2d_kernel_wrapper(int b, int n, int m, int k, const float *query_xyz, const float *input_xyz, int64_t *indices);
void k_nearest_neighbor_3d_kernel_wrapper(int b, int n, int m, int k, const float *query_xyz, const float *input_xyz, int64_t *indices);

torch::Tensor k_nearest_neighbor_cuda(torch::Tensor input_xyz, torch::Tensor query_xyz, int k) {
	TORCH_CHECK(input_xyz.is_contiguous(), "input_xyz must be a contiguous tensor");
	TORCH_CHECK(input_xyz.is_cuda(), "input_xyz must be a CUDA tensor");
	TORCH_CHECK(input_xyz.scalar_type() == torch::ScalarType::Float, "input_xyz must be a float tensor");

	TORCH_CHECK(query_xyz.is_contiguous(), "query_xyz must be a contiguous tensor");
	TORCH_CHECK(query_xyz.is_cuda(), "query_xyz must be a CUDA tensor");
	TORCH_CHECK(query_xyz.scalar_type() == torch::ScalarType::Float, "query_xyz must be a float tensor");

	int batch_size = query_xyz.size(0), n_queries = query_xyz.size(1), n_inputs = input_xyz.size(1), n_dim = query_xyz.size(2);
	torch::Tensor indices = torch::zeros({batch_size, n_queries, k}, torch::device(query_xyz.device()).dtype(torch::ScalarType::Long));

	if (n_dim == 2)
		k_nearest_neighbor_2d_kernel_wrapper(batch_size, n_queries, n_inputs, k, query_xyz.data_ptr<float>(), input_xyz.data_ptr<float>(), indices.data_ptr<int64_t>());
	else
		k_nearest_neighbor_3d_kernel_wrapper(batch_size, n_queries, n_inputs, k, query_xyz.data_ptr<float>(), input_xyz.data_ptr<float>(), indices.data_ptr<int64_t>());

	return indices;
}

#ifdef TORCH_EXTENSION_NAME
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
	m.def("_k_nearest_neighbor_cuda", &k_nearest_neighbor_cuda, "CUDA implementation of KNN");
}
#endif