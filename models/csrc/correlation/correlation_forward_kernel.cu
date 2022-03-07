#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#define BLOCK_SIZE 32

__forceinline__ __device__ float warpReduceSum(float val) {
	for (int offset = 16; offset > 0; offset /= 2)
		val += __shfl_down_sync(0xffffffff, val, offset);
	return val;
}

__global__ void correlation_forward_kernel(float* __restrict__ output,
										   const float* __restrict__ input1,
										   const float* __restrict__ input2,
										   int in_channels, int height, int width, int max_displacement) {
	int n = blockIdx.x, y1 = blockIdx.y, x1 = blockIdx.z;

	int displacement_size = 2 * max_displacement + 1;
	int out_channels = displacement_size * displacement_size;

	int in_stride0 = height * width * in_channels;
	int in_stride1 = width * in_channels;
	int in_stride2 = in_channels;

	int out_stride0 = out_channels * height * width;
	int out_stride1 = height * width;
	int out_stride2 = width;

	for (int tx = -max_displacement; tx <= max_displacement; ++tx)
		for (int ty = -max_displacement; ty <= max_displacement; ++ty) {
			int x2 = x1 + ty, y2 = y1 + tx;
			if (x2 < 0 || y2 < 0 || x2 >= width || y2 >= height) continue;
			
			float sum = 0.0f;
			for (int c = threadIdx.x; c < in_channels; c += BLOCK_SIZE) {
				int idx1 = n * in_stride0 + y1 * in_stride1 + x1 * in_stride2 + c;
				int idx2 = n * in_stride0 + y2 * in_stride1 + x2 * in_stride2 + c;
				sum += input1[idx1] * input2[idx2];
			}

			__syncthreads();
			sum = warpReduceSum(sum);

			if (threadIdx.x == 0) {
				int tc = (tx + max_displacement) * displacement_size + (ty + max_displacement);
				int idx = n * out_stride0 + tc * out_stride1 + y1 * out_stride2 + x1;
				output[idx] = sum / in_channels;
			}
		}
}

void correlation_forward_kernel_wrapper(float* output, const float* input1, const float* input2,
								        int n_batches, int in_channels, int height, int width, int max_displacement) {
	dim3 number_of_blocks(n_batches, height, width);
	correlation_forward_kernel<<<number_of_blocks, BLOCK_SIZE>>>(output, input1, input2, in_channels, height, width, max_displacement);
}
