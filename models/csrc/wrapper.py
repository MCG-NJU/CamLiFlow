import torch
import torch.nn.functional

try:
    from ._correlation_cuda import _correlation_forward_cuda
    from ._correlation_cuda import _correlation_backward_cuda
    from ._furthest_point_sampling_cuda import _furthest_point_sampling_cuda
    from ._k_nearest_neighbor_cuda import _k_nearest_neighbor_cuda
except ImportError as e:
    _correlation_forward_cuda = None
    _correlation_backward_cuda = None
    _furthest_point_sampling_cuda = None
    _k_nearest_neighbor_cuda = None
    print('Failed to load one or more CUDA extensions, performance may be hurt.')
    print('Error message:', e)


class CorrelationFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input1, input2, max_displacement):
        ctx.save_for_backward(input1, input2)
        ctx.max_displacement = max_displacement
        assert callable(_correlation_forward_cuda)
        return _correlation_forward_cuda(input1, input2, max_displacement)

    @staticmethod
    def backward(ctx, grad_output):
        input1, input2 = ctx.saved_tensors

        assert callable(_correlation_backward_cuda)
        grad_input1, grad_input2 = _correlation_backward_cuda(
            grad_output, input1, input2, ctx.max_displacement
        )
        grad_input1 = grad_input1.permute(0, 2, 3, 1).contiguous()
        grad_input2 = grad_input2.permute(0, 2, 3, 1).contiguous()

        return grad_input1, grad_input2, None


def squared_distance(xyz1: torch.Tensor, xyz2: torch.Tensor):
    """
    Calculate the Euclidean squared distance between every two points.
    :param xyz1: the 1st set of points, [batch_size, n_points_1, 3]
    :param xyz2: the 2nd set of points, [batch_size, n_points_2, 3]
    :return: squared distance between every two points, [batch_size, n_points_1, n_points_2]
    """
    assert xyz1.shape[-1] == xyz2.shape[-1] and xyz1.shape[-1] <= 3  # assert channel_last
    batch_size, n_points1, n_points2 = xyz1.shape[0], xyz1.shape[1], xyz2.shape[1]
    dist = -2 * torch.matmul(xyz1, xyz2.permute(0, 2, 1))
    dist += torch.sum(xyz1 ** 2, -1).view(batch_size, n_points1, 1)
    dist += torch.sum(xyz2 ** 2, -1).view(batch_size, 1, n_points2)
    return dist


def correlation2d(input1: torch.Tensor, input2: torch.Tensor, max_displacement: int, cpp_impl=True):
    def _correlation_py(_input1, _input2, _max_displacement):
        height, width = _input1.shape[2:]
        _input2 = torch.nn.functional.pad(_input2, [_max_displacement] * 4)
        cost_volumes = []
        for i in range(2 * _max_displacement + 1):
            for j in range(2 * _max_displacement + 1):
                cost_volume = _input1 * _input2[:, :, i:(i + height), j:(j + width)]
                cost_volume = torch.mean(cost_volume, 1, keepdim=True)
                cost_volumes.append(cost_volume)
        return torch.cat(cost_volumes, 1)

    if cpp_impl and callable(_correlation_forward_cuda) and callable(_correlation_backward_cuda) and input1.is_cuda and input2.is_cuda:
        input1 = input1.permute(0, 2, 3, 1).contiguous().float()
        input2 = input2.permute(0, 2, 3, 1).contiguous().float()
        return CorrelationFunction.apply(input1, input2, max_displacement)
    else:
        return _correlation_py(input1, input2, max_displacement)


def furthest_point_sampling(xyz: torch.Tensor, n_samples: int, cpp_impl=True):
    """
    Perform furthest point sampling on a set of points.
    :param xyz: a set of points, [batch_size, n_points, 3]
    :param n_samples: number of samples, int
    :param cpp_impl: whether to use the CUDA C++ implementation of furthest-point-sampling
    :return: indices of sampled points, [batch_size, n_samples]
    """
    def _furthest_point_sampling_py(_xyz: torch.Tensor, _n_samples: int):
        batch_size, n_points, _ = _xyz.shape
        farthest_indices = torch.zeros(batch_size, _n_samples, dtype=torch.int64, device=_xyz.device)
        distances = torch.ones(batch_size, n_points, device=_xyz.device) * 1e10
        batch_indices = torch.arange(batch_size, dtype=torch.int64, device=_xyz.device)
        curr_farthest_idx = torch.zeros(batch_size, dtype=torch.int64, device=_xyz.device)
        for i in range(_n_samples):
            farthest_indices[:, i] = curr_farthest_idx
            curr_farthest = _xyz[batch_indices, curr_farthest_idx, :].view(batch_size, 1, 3)
            new_distances = torch.sum((_xyz - curr_farthest) ** 2, -1)
            mask = new_distances < distances
            distances[mask] = new_distances[mask]
            curr_farthest_idx = torch.max(distances, -1)[1]
        return farthest_indices

    assert xyz.shape[2] == 3 and xyz.shape[1] > n_samples

    if cpp_impl and callable(_furthest_point_sampling_cuda) and xyz.is_cuda:
        return _furthest_point_sampling_cuda(xyz.contiguous(), n_samples).to(torch.int64)
    else:
        return _furthest_point_sampling_py(xyz, n_samples).to(torch.int64)


def k_nearest_neighbor(input_xyz: torch.Tensor, query_xyz: torch.Tensor, k: int, cpp_impl=True):
    """
    Calculate k-nearest neighbor for each query.
    :param input_xyz: a set of points, [batch_size, n_points, 3] or [batch_size, 3, n_points]
    :param query_xyz: a set of centroids, [batch_size, n_queries, 3] or [batch_size, 3, n_queries]
    :param k: int
    :param cpp_impl: whether to use the CUDA C++ implementation of k-nearest-neighbor
    :return: indices of k-nearest neighbors, [batch_size, n_queries, k]
    """
    def _k_nearest_neighbor_py(_input_xyz: torch.Tensor, _query_xyz: torch.Tensor, _k: int):
        dists = squared_distance(_query_xyz, _input_xyz)
        return dists.topk(_k, dim=2, largest=False).indices.to(torch.long)

    if input_xyz.shape[1] <= 3:  # channel_first to channel_last
        assert query_xyz.shape[1] == input_xyz.shape[1]
        input_xyz = input_xyz.transpose(1, 2).contiguous()
        query_xyz = query_xyz.transpose(1, 2).contiguous()

    if cpp_impl and callable(_k_nearest_neighbor_cuda) and input_xyz.is_cuda and query_xyz.is_cuda:
        return _k_nearest_neighbor_cuda(input_xyz.contiguous(), query_xyz.contiguous(), k)
    else:
        return _k_nearest_neighbor_py(input_xyz, query_xyz, k)
