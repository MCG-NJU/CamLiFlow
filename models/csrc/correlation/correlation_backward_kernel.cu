#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void correlation_backward_input1_kernel(float* __restrict__ grad_input1,
												   const float* __restrict__ grad_output,
												   const float* __restrict__ input2,
												   int in_channels, int height, int width, int max_displacement) {
	int n = blockIdx.x, y1 = blockIdx.y, x1 = blockIdx.z, c = threadIdx.x;

	int displacement_size = 2 * max_displacement + 1;
	int out_channels = displacement_size * displacement_size;

	int in_stride0 = height * width * in_channels;
	int in_stride1 = width * in_channels;
	int in_stride2 = in_channels;

	int out_stride0 = out_channels * height * width;
	int out_stride1 = height * width;
	int out_stride2 = width;

	int grad_in_stride0 = in_channels * height* width;
	int grad_in_stride1 = height * width;
	int grad_in_stride2 = width;

	float sum1 = 0;
	for (int tx = -max_displacement; tx <= max_displacement; ++tx)
		for (int ty = -max_displacement; ty <= max_displacement; ++ty) {
			int x2 = x1 + ty, y2 = y1 + tx;
			if (x2 < 0 || y2 < 0 || x2 >= width || y2 >= height) continue;
			int tc = (tx + max_displacement) * displacement_size + (ty + max_displacement);
			int idx = n * out_stride0 + tc * out_stride1 + y1 * out_stride2 + x1;
			int idx2 = n * in_stride0 + y2 * in_stride1 + x2 * in_stride2 + c;
			sum1 += grad_output[idx] * input2[idx2];
		}

	int idx1 = n * grad_in_stride0 + c * grad_in_stride1 + y1 * grad_in_stride2 + x1;
	grad_input1[idx1] = sum1 / in_channels;
}

__global__ void correlation_backward_input2_kernel(float* __restrict__ grad_input2,
												   const float* __restrict__ grad_output,
												   const float* __restrict__ input1,
												   int in_channels, int height, int width, int max_displacement) {
	int n = blockIdx.x, y2 = blockIdx.y, x2 = blockIdx.z, c = threadIdx.x;

	int displacement_size = 2 * max_displacement + 1;
	int out_channels = displacement_size * displacement_size;

	int in_stride0 = height * width * in_channels;
	int in_stride1 = width * in_channels;
	int in_stride2 = in_channels;

	int out_stride0 = out_channels * height * width;
	int out_stride1 = height * width;
	int out_stride2 = width;

	int grad_in_stride0 = in_channels * height* width;
	int grad_in_stride1 = height * width;
	int grad_in_stride2 = width;

	float sum2 = 0;
	for (int tx = -max_displacement; tx <= max_displacement; ++tx)
		for (int ty = -max_displacement; ty <= max_displacement; ++ty) {
			int x1 = x2 - ty, y1 = y2 - tx;
			if (x1 < 0 || y1 < 0 || x1 >= width || y1 >= height) continue;
			int tc = (tx + max_displacement) * displacement_size + (ty + max_displacement);
			int idx = n * out_stride0 + tc * out_stride1 + y1 * out_stride2 + x1;
			int idx1 = n * in_stride0 + y1 * in_stride1 + x1 * in_stride2 + c;
			sum2 += grad_output[idx] * input1[idx1];
		}

	int idx2 = n * grad_in_stride0 + c * grad_in_stride1 + y2 * grad_in_stride2 + x2;
	grad_input2[idx2] = sum2 / in_channels;
}

void correlation_backward_kernel_wrapper(const float* grad_output,
										 float* grad_input1, float* grad_input2,
										 const float* input1, const float* input2,
										 int n_batches, int in_channels, int height, int width, int max_displacement) {
	dim3 totalBlocksCorr(n_batches, height, width);
	correlation_backward_input1_kernel<<<totalBlocksCorr, in_channels>>>(
		grad_input1, grad_output, input2,
		in_channels, height, width, max_displacement
	);
	correlation_backward_input2_kernel<<<totalBlocksCorr, in_channels>>>(
		grad_input2, grad_output, input1,
		in_channels, height, width, max_displacement
	);
}
