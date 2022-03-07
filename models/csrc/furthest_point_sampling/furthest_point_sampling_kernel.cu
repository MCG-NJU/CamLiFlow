#include <stdint.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define argmaxReduceMacro(arr, arr_idx, idx1, idx2) {\
	if (arr[idx1] <= arr[idx2]) {\
        arr[idx1] = arr[idx2];\
		arr_idx[idx1] = arr_idx[idx2];\
	}\
}

template <unsigned int block_size>
__device__ void warpReduce(volatile float* max_values, volatile int* max_values_idx, int tid) {
	if (block_size >= 64) argmaxReduceMacro(max_values, max_values_idx, tid, tid + 32);
	if (block_size >= 32) argmaxReduceMacro(max_values, max_values_idx, tid, tid + 16);
	if (block_size >= 16) argmaxReduceMacro(max_values, max_values_idx, tid, tid + 8);
	if (block_size >= 8) argmaxReduceMacro(max_values, max_values_idx, tid, tid + 4);
	if (block_size >= 4) argmaxReduceMacro(max_values, max_values_idx, tid, tid + 2);
	if (block_size >= 2) argmaxReduceMacro(max_values, max_values_idx, tid, tid + 1);
}

template <unsigned int block_size>
__device__ int64_t argmax(float* max_values, int* max_values_idx) {
	int tid = threadIdx.x;
	if (block_size >= 1024) { if (tid < 512) argmaxReduceMacro(max_values, max_values_idx, tid, tid + 512); __syncthreads(); }
	if (block_size >= 512) { if (tid < 256) argmaxReduceMacro(max_values, max_values_idx, tid, tid + 256); __syncthreads(); }
	if (block_size >= 256) { if (tid < 128) argmaxReduceMacro(max_values, max_values_idx, tid, tid + 128); __syncthreads(); }
	if (block_size >= 128) { if (tid < 64) argmaxReduceMacro(max_values, max_values_idx, tid, tid + 64); __syncthreads(); }
	if (tid < 32) warpReduce<block_size>(max_values, max_values_idx, tid);
	__syncthreads();
	return max_values_idx[0];
}

template <unsigned int block_size>
__global__ void furthest_point_sampling_kernel(float* __restrict__ batched_points_xyz,
											   float* __restrict__ batched_dists_temp,
											   int n_points, int n_samples,
											   int64_t* __restrict__ batched_furthest_indices) {
	int bid = blockIdx.x, tid = threadIdx.x;

	float* __restrict__ points_xyz = batched_points_xyz + bid * n_points * 3;
	float* __restrict__ dists_temp = batched_dists_temp + bid * n_points;
	int64_t* __restrict__ furthest_indices = batched_furthest_indices + bid * n_samples;
	
	__shared__ float max_dists[block_size];
	__shared__ int max_dists_idx[block_size];

	int64_t curr_furthest_idx = 0;
	for (int s = 0; s < n_samples; s++) {
		if (tid == 0) furthest_indices[s] = curr_furthest_idx;

		float x1 = points_xyz[curr_furthest_idx * 3 + 0];
		float y1 = points_xyz[curr_furthest_idx * 3 + 1];
		float z1 = points_xyz[curr_furthest_idx * 3 + 2];

		float local_max_dist = -1;
		int local_max_dist_idx = 0;

		for (int i = tid; i < n_points; i += block_size) {
			float x2 = points_xyz[i * 3 + 0];
			float y2 = points_xyz[i * 3 + 1];
			float z2 = points_xyz[i * 3 + 2];
			float d = (x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1) + (z2 - z1) * (z2 - z1);
			float new_dist = min(dists_temp[i], d);
			if (new_dist > local_max_dist) {
				local_max_dist = new_dist;
				local_max_dist_idx = i;
			}
			dists_temp[i] = new_dist;
		}

		max_dists[tid] = local_max_dist;
		max_dists_idx[tid] = local_max_dist_idx;

		__syncthreads();

		curr_furthest_idx = argmax<block_size>(max_dists, max_dists_idx);
	}
}

void furthest_point_sampling_kernel_wrapper(float* batched_points_xyz, float* batched_dists_temp,
											int n_batch, int n_points, int n_samples,
											int64_t* batched_furthest_indices) {
	furthest_point_sampling_kernel<1024> <<<n_batch, 1024>>> (batched_points_xyz, batched_dists_temp, n_points, n_samples, batched_furthest_indices);
}