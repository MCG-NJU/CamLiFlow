#include <stdint.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define THREADS_PER_BLOCK 256
#define DIVUP(m, n) ((m) / (n) + ((m) % (n) > 0))

__global__ void k_nearest_neighbor_2d_kernel(int b, int n, int m, int k,
											 const float *__restrict__ query_xyz,
											 const float *__restrict__ input_xyz,
											 int64_t *__restrict__ indices) {
	int bs_idx = blockIdx.y;
	int pt_idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (bs_idx >= b || pt_idx >= n) return;

	query_xyz += bs_idx * n * 2 + pt_idx * 2;
	input_xyz += bs_idx * m * 2;
	indices += bs_idx * n * k + pt_idx * k;

	float ux = query_xyz[0];
	float uy = query_xyz[1];

	// initialize
	float nn_dists[32]; int nn_indices[32];
	for (int i = 0; i < 32; i++) {
		nn_dists[i] = 1e9;
		nn_indices[i] = 0;
	}

	for (int idx = 0; idx < m; idx++) {
		float x = input_xyz[idx * 2 + 0];
		float y = input_xyz[idx * 2 + 1];
		float d = (ux - x) * (ux - x) + (uy - y) * (uy - y);
		if (d > nn_dists[k - 1]) continue;

		int j = min(idx, k - 1);
		while (j > 0 && nn_dists[j - 1] > d) {
			nn_dists[j] = nn_dists[j - 1];
			nn_indices[j] = nn_indices[j - 1];
			j--;
		}

		nn_dists[j] = d;
		nn_indices[j] = idx;
	}

	for (int i = 0; i < k; i++)
		indices[i] = nn_indices[i];
}

__global__ void k_nearest_neighbor_3d_kernel(int b, int n, int m, int k,
										  const float *__restrict__ query_xyz,
										  const float *__restrict__ input_xyz,
										  int64_t *__restrict__ indices) {
	int bs_idx = blockIdx.y;
	int pt_idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (bs_idx >= b || pt_idx >= n) return;

	query_xyz += bs_idx * n * 3 + pt_idx * 3;
	input_xyz += bs_idx * m * 3;
	indices += bs_idx * n * k + pt_idx * k;

	float ux = query_xyz[0];
	float uy = query_xyz[1];
	float uz = query_xyz[2];

	// initialize
	float nn_dists[32]; int nn_indices[32];
	for (int i = 0; i < 32; i++) {
		nn_dists[i] = 1e9;
		nn_indices[i] = 0;
	}

	for (int idx = 0; idx < m; idx++) {
		float x = input_xyz[idx * 3 + 0];
		float y = input_xyz[idx * 3 + 1];
		float z = input_xyz[idx * 3 + 2];
		float d = (ux - x) * (ux - x) + (uy - y) * (uy - y) + (uz - z) * (uz - z);
		if (d > nn_dists[k - 1]) continue;

		int j = min(idx, k - 1);
		while (j > 0 && nn_dists[j - 1] > d) {
			nn_dists[j] = nn_dists[j - 1];
			nn_indices[j] = nn_indices[j - 1];
			j--;
		}

		nn_dists[j] = d;
		nn_indices[j] = idx;
	}

	for (int i = 0; i < k; i++)
		indices[i] = nn_indices[i];
}

void k_nearest_neighbor_2d_kernel_wrapper(int b, int n, int m, int k,
										  const float *query_xyz,
										  const float *input_xyz,
										  int64_t *indices) {
	dim3 blocks(DIVUP(n, THREADS_PER_BLOCK), b);  // blockIdx.x(col), blockIdx.y(row)
	dim3 threads(THREADS_PER_BLOCK);
	k_nearest_neighbor_2d_kernel<<<blocks, threads>>>(b, n, m, k, query_xyz, input_xyz, indices);
}

void k_nearest_neighbor_3d_kernel_wrapper(int b, int n, int m, int k,
									   const float *query_xyz,
									   const float *input_xyz,
									   int64_t *indices) {
	dim3 blocks(DIVUP(n, THREADS_PER_BLOCK), b);  // blockIdx.x(col), blockIdx.y(row)
	dim3 threads(THREADS_PER_BLOCK);
	k_nearest_neighbor_3d_kernel<<<blocks, threads>>>(b, n, m, k, query_xyz, input_xyz, indices);
}
