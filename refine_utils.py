import torch
import numpy as np
import cv2

# copy from https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py
seg_colors = [
    (0, 0, 0),          # id 0, unlabeled, void
    (0, 0, 0),          # id 1, ego vehicle, void
    (0, 0, 0),          # id 2, rectification border, void
    (0, 0, 0),          # id 3, out of roi, void
    (0, 0, 0),          # id 4, static, void
    (111, 74, 0),       # id 5, dynamic, void
    (81, 0, 81),        # id 6, ground, void
    (128, 64, 128),     # id 7, road, flat
    (244, 35, 232),     # id 8, sidewalk, flat
    (250, 170, 160),    # id 9, parking, flat
    (230, 150, 140),    # id 10, rail track, flat
    (70, 70, 70),       # id 11, building, construction
    (102, 102, 156),    # id 12, wall, construction
    (190, 153, 153),    # id 13, fence, construction
    (180, 165, 180),    # id 14, guard rail construction
    (150, 100, 100),    # id 15, bridge, construction
    (150, 120, 90),     # id 16, tunnel, construction
    (153, 153, 153),    # id 17, pole, object
    (153, 153, 153),    # id 18, polegroup, object
    (250, 170, 30),     # id 19, traffic light, object
    (220, 220, 0),      # id 20, traffic sign, object
    (107, 142, 35),     # id 21, vegetation, nature
    (152, 251, 152),    # id 22, terrain, nature
    (70, 130, 180),     # id 23, sky, sky
    (220, 20, 60),      # id 24, person, human
    (255, 0, 0),        # id 25, rider, human
    (0, 0, 142),        # id 26, car, vehicle
    (0, 0, 70),         # id 27, truck, vehicle
    (0, 60, 100),       # id 28, bus, vehicle
    (0, 0, 90),         # id 29, caravan, vehicle
    (0, 0, 110),        # id 30, trailer, vehicle
    (0, 80, 100),       # id 31, train, vehicle
    (0, 0, 230),        # id 32, motorcycle, vehicle
    (119, 11, 32),      # id 33, bicycle, vehicle
    (0, 0, 142),        # id 34, license plate, vehicle
]


# borrowed from https://github.com/gengshan-y/rigidmask/blob/main/utils/dydepth.py#L213
def triangulation(disp, xcoord, ycoord, bl=1, fl=450, cx=479.5, cy=269.5):
    mask = (disp <= 0).flatten()
    depth = bl * fl / (disp)  # 450px->15mm focal length
    X = (xcoord - cx) * depth / fl
    Y = (ycoord - cy) * depth / fl
    Z = depth
    P = np.concatenate((X[np.newaxis], Y[np.newaxis], Z[np.newaxis]), 0).reshape(3, -1)
    P = np.concatenate((P, np.ones((1, P.shape[-1]))), 0)
    P[:, mask] = 0
    return P


# borrowed from https://github.com/gengshan-y/rigidmask/blob/main/utils/dydepth.py#L224
def midpoint_triangulate(x, cam):
    """
    Args:
        x:   Set of 2D points in homogeneous coords, (3 x n x N) matrix
        cam: Collection of n objects, each containing member variables
                 cam.P - 3x4 camera matrix [0]
                 cam.R - 3x3 rotation matrix [1]
                 cam.T - 3x1 translation matrix [2]
    Returns:
        midpoint: 3D point in homogeneous coords, (4 x 1) matrix
    """
    n = len(cam)  # No. of cameras
    N = x.shape[-1]

    I = np.eye(3)  # 3x3 identity matrix
    A = np.zeros((3, n))
    B = np.zeros((3, n, N))
    sigma2 = np.zeros((3, N))

    for i in range(n):
        a = -np.linalg.inv(cam[i][:3, :3]).dot(cam[i][:3, -1:])  # ith camera position #
        A[:, i, None] = a
        if i == 0:
            b = np.linalg.pinv(cam[i][:3, :3]).dot(x[:, i])  # Directional vector # 4, N
        else:
            b = np.linalg.pinv(cam[i]).dot(x[:, i])  # Directional vector # 4, N
            b = b / b[3:]
            b = b[:3, :] - a  # 3,N
        b = b / np.linalg.norm(b, 2, 0)[np.newaxis]
        B[:, i, :] = b

        sigma2 = sigma2 + b * (b.T.dot(a).reshape(-1, N))  # 3,N

    Bo = B.transpose([2, 0, 1])
    Bt = B.transpose([2, 1, 0])

    Bo = torch.DoubleTensor(Bo)
    Bt = torch.DoubleTensor(Bt)
    A = torch.DoubleTensor(A)
    sigma2 = torch.DoubleTensor(sigma2)
    I = torch.DoubleTensor(I)

    BoBt = torch.matmul(Bo, Bt)
    C = (n * I)[np.newaxis] - BoBt  # N,3,3
    Cinv = C.inverse()
    sigma1 = torch.sum(A, dim=1)[:, None]
    m1 = I[np.newaxis] + torch.matmul(BoBt, Cinv)
    m2 = torch.matmul(Cinv, sigma2.T[:, :, np.newaxis])
    midpoint = (1 / n) * torch.matmul(m1, sigma1[np.newaxis]) - m2

    midpoint = np.asarray(midpoint)
    return midpoint[:, :, 0].T, np.asarray(Bo)


# borrowed from https://github.com/gengshan-y/rigidmask/blob/main/utils/dydepth.py#L277
def register_disp_fast(id_flow, id_mono, mask, inlier_th=0.01, niters=100):
    """ 
    input: disp_flow, disp_mono, mask
    output: inlier_mask, registered
    register up-to-scale rough depth to motion-based depth
    """
    shape = id_mono.shape
    id_mono = id_mono.flatten()
    disp_flow = id_flow[mask]  # register to flow with mono
    disp_mono = id_mono[mask]

    num_samp = min(3000, len(disp_flow))
    np.random.seed(0)
    submask = np.random.choice(range(len(disp_flow)), num_samp)
    disp_flow = disp_flow[submask]
    disp_mono = disp_mono[submask]

    n = len(disp_flow)
    sample_size = niters
    rand_idx = np.random.choice(range(n), sample_size)
    scale_cand = (disp_flow / disp_mono)[rand_idx]
    dis_cand = np.abs(np.log(disp_mono[:, np.newaxis] * scale_cand[np.newaxis]) - np.log(disp_flow[:, np.newaxis]))

    rank_metric = (dis_cand < inlier_th).sum(0)
    scale_idx = np.argmax(rank_metric)
    scale = scale_cand[scale_idx]

    dis = np.abs(np.log(disp_mono * scale) - np.log(disp_flow))
    ninliers = (dis < inlier_th).sum() / n
    registered_flow = (id_flow.reshape(shape)) / scale

    return registered_flow, scale, ninliers


# borrowed from https://github.com/gengshan-y/rigidmask/blob/main/models/submodule.py#L661
def F_ngransac(hp0, hp1, Ks, rand, unc_occ, iters=1000, cv=False, Kn=None):
    if Kn is None:
        Kn = Ks
    import cv2

    b = hp1.shape[0]
    hp0_cpu = np.asarray(hp0.cpu())
    hp1_cpu = np.asarray(hp1.cpu())
    if not rand:
        fmask = np.ones(hp0.shape[1]).astype(bool)
        rand_seed = 0
    else:
        fmask = np.random.choice([True, False], size=hp0.shape[1], p=[0.1, 0.9])
        rand_seed = np.random.randint(0, 1000)  # random seed to by used in C++
    hp0 = Ks.inverse().matmul(hp0.permute(0, 2, 1)).permute(0, 2, 1)
    hp1 = Kn.inverse().matmul(hp1.permute(0, 2, 1)).permute(0, 2, 1)
    ratios = torch.zeros(hp0[:1, :, :1].shape)
    probs = torch.Tensor(np.ones(fmask.sum())) / fmask.sum()
    probs = probs[np.newaxis, :, np.newaxis]

    Es = torch.zeros((b, 3, 3)).float()  # estimated model
    rot = torch.zeros((b, 3)).float()  # estimated model
    trans = torch.zeros((b, 3)).float()  # estimated model
    out_model = torch.zeros((3, 3)).float()  # estimated model
    out_inliers = torch.zeros(probs.size())  # inlier mask of estimated model
    out_gradients = torch.zeros(probs.size())  # gradient tensor (only used during training)

    for i in range(b):
        pts1 = hp0[i:i + 1, fmask, :2].cpu()
        pts2 = hp1[i:i + 1, fmask, :2].cpu()
        # create data tensor of feature coordinates and matching ratios
        correspondences = torch.cat([pts1, pts2, ratios], dim=2)
        correspondences = correspondences.permute(2, 1, 0)

        if cv == True:
            E, ffmask = cv2.findEssentialMat(np.asarray(pts1[0]), np.asarray(pts2[0]), np.eye(3), cv2.FM_RANSAC,
                                             threshold=0.0001)
            ffmask = ffmask[:, 0]
            Es[i] = torch.Tensor(E)
        else:
            import ngransac
            incount = ngransac.find_essential_mat(correspondences, probs, rand_seed, iters, 0.0001, out_model,
                                                  out_inliers, out_gradients)
            Es[i] = out_model
            E = np.asarray(out_model)
            maskk = np.asarray(out_inliers[0, :, 0])
            ffmask = fmask.copy()
            ffmask[fmask] = maskk
        K1 = np.asarray(Kn[i].cpu())
        K0 = np.asarray(Ks[i].cpu())
        R1, R2, T = cv2.decomposeEssentialMat(E)
        for rott in [(R1, T), (R2, T), (R1, -T), (R2, -T)]:
            if testEss(K0, K1, rott[0], rott[1], hp0_cpu[0, ffmask].T, hp1_cpu[i, ffmask].T):
                R01 = rott[0].T
                t10 = -R01.dot(rott[1][:, 0])
        if 't10' not in locals():
            t10 = np.asarray([0, 0, 1])
            R01 = np.eye(3)
        rot[i] = torch.Tensor(cv2.Rodrigues(R01)[0][:, 0]).cuda()
        trans[i] = torch.Tensor(t10).cuda()

    return rot, trans, Es


# borrowed from https://github.com/gengshan-y/rigidmask/blob/main/utils/dydepth.py#L321
def testEss(K0, K1, R, T, p1, p2):
    testP = cv2.triangulatePoints(K0.dot(np.concatenate((np.eye(3), np.zeros((3, 1))), -1)),
                                  K1.dot(np.concatenate((R, T), -1)),
                                  p1[:2], p2[:2])
    Z1 = testP[2, :] / testP[-1, :]
    Z2 = (R.dot(Z1 * np.linalg.inv(K0).dot(p1)) + T)[-1, :]
    if ((Z1 > 0).sum() > (Z1 <= 0).sum()) and ((Z2 > 0).sum() > (Z2 <= 0).sum()):
        return True
    else:
        return False


# borrowed from https://github.com/gengshan-y/rigidmask/blob/main/utils/dydepth.py#L334
def pose_estimate(K0, K1, hp0, hp1, strict_mask, rot, th=0.0001):
    # epipolar geometry
    tmphp0 = hp0[:, strict_mask]
    tmphp1 = hp1[:, strict_mask]
    num_samp = min(3000, tmphp0.shape[1])
    submask = np.random.choice(range(tmphp0.shape[1]), num_samp)
    tmphp0 = tmphp0[:, submask]
    tmphp1 = tmphp1[:, submask]

    rotx, transx, Ex = F_ngransac(torch.Tensor(tmphp0.T[np.newaxis]).cuda(),
                                  torch.Tensor(tmphp1.T[np.newaxis]).cuda(),
                                  torch.Tensor(K0[np.newaxis]).cuda(),
                                  False, 0,
                                  Kn=torch.Tensor(K1[np.newaxis]).cuda())
    R01 = cv2.Rodrigues(np.asarray(rotx[0]))[0]
    T01 = np.asarray(transx[0])
    E = np.asarray(Ex[0])

    R1, R2, T = cv2.decomposeEssentialMat(E)
    for rott in [(R1, T), (R2, T), (R1, -T), (R2, -T)]:
        if testEss(K0, K1, rott[0], rott[1], tmphp0, tmphp1):
            R01 = rott[0].T
            T01 = -R01.dot(rott[1][:, 0])
    if not 'T01' in locals():
        T01 = np.asarray([0, 0, 1])
        R01 = np.eye(3)

    # compensate R
    H01 = K0.dot(R01).dot(np.linalg.inv(K1))  # plane at infinity
    comp_hp1 = H01.dot(hp1)
    comp_hp1 = comp_hp1 / comp_hp1[-1:]
    return R01, T01, H01, comp_hp1, E


# https://github.com/gengshan-y/rigidmask/blob/main/utils/dydepth.py#L393
def evaluate_tri(t10, R01, K0, K1, hp0, hp1, disp0, bl, inlier_th=0.1, select_th=0.4, valid_mask=None):
    if valid_mask is not None:
        hp0 = hp0[:, valid_mask]
        hp1 = hp1[:, valid_mask]
        disp0 = disp0.flatten()[valid_mask]
    # triangluation
    cams = [K0.dot(np.concatenate((np.eye(3), np.zeros((3, 1))), -1)),
            K1.dot(np.concatenate((R01.T, -R01.T.dot(t10[:, np.newaxis])), -1))]
    P_pred, _ = midpoint_triangulate(np.concatenate([hp0[:, np.newaxis], hp1[:, np.newaxis]], 1), cams)
    idepth_p3d = np.clip(K0[0, 0] * bl / P_pred[2], 1e-6, np.inf)

    # discard points with small disp
    entmask = np.logical_and(idepth_p3d > 1e-12, ~np.isinf(idepth_p3d))
    entmask_tmp = entmask[entmask].copy()
    entmask_tmp[np.argsort(-idepth_p3d[entmask])[entmask.sum() // 2:]] = False  # remove sky
    entmask[entmask] = entmask_tmp
    med = np.median(idepth_p3d[entmask])
    entmask = np.logical_and(entmask, np.logical_and(idepth_p3d > med / 5., idepth_p3d < med * 5))
    if entmask.sum() < 10:
        return None, None, None
    registered_p3d, scale, ninliers = register_disp_fast(idepth_p3d, disp0, entmask,
                                                         inlier_th=inlier_th, niters=100)

    disp_ratio = np.abs(np.log(registered_p3d.flatten() / disp0.flatten()))
    agree_mask = disp_ratio < np.log(select_th)
    rank = np.argsort(disp_ratio)
    return agree_mask, t10 * scale, rank


def mod_flow(bg_mask, disp, disp_change, flow, K0, K1, bl, occ_mask, parallax_th=8):
    # prepare data
    flow = flow.copy()
    h, w = flow.shape[:2]
    x0, y0 = np.meshgrid(range(w), range(h))
    x0 = x0.astype(np.float32)
    y0 = y0.astype(np.float32)
    x1 = x0 + flow[:, :, 0]
    y1 = y0 + flow[:, :, 1]
    hp0 = np.concatenate((x0[np.newaxis], y0[np.newaxis], np.ones(x1.shape)[np.newaxis]), 0).reshape((3, -1))
    hp1 = np.concatenate((x1[np.newaxis], y1[np.newaxis], np.ones(x1.shape)[np.newaxis]), 0).reshape((3, -1))

    # use bg + valid pixels to compute R/t
    valid_mask = np.logical_and(disp > 0, np.logical_and(bg_mask, occ_mask)).flatten()

    R01, T01, _, comp_hp1, _ = pose_estimate(K0, K1, hp0, hp1, valid_mask, [0, 0, 0])

    parallax = np.transpose((comp_hp1[:2] - hp0[:2]), [1, 0]).reshape(x1.shape + (2,))
    parallax_mag = np.linalg.norm(parallax[:, :, :2], 2, 2)

    reg_flow_P = triangulation(disp, x0, y0, bl=bl, fl=K0[0, 0], cx=K0[0, 2], cy=K0[1, 2])[:3]

    is_static = parallax_mag[bg_mask].mean() < parallax_th

    # modify motion fields
    if not is_static:
        aligned_mask, T01_c, ranked_p = evaluate_tri(T01, R01, K0, K1, hp0, hp1, disp, bl, inlier_th=0.01,
                                                     select_th=1.2, valid_mask=valid_mask)
        # PnP refine
        aligned_mask[ranked_p[50000:]] = False
        tmp = valid_mask.copy()
        tmp[tmp] = aligned_mask
        aligned_mask = tmp
        _, rvec, T01 = cv2.solvePnP(reg_flow_P.T[aligned_mask.flatten(), np.newaxis],
                                    hp1[:2].T[aligned_mask.flatten(), np.newaxis], K0, 0,
                                    flags=cv2.SOLVEPNP_DLS)
        _, rvec, T01, = cv2.solvePnP(reg_flow_P.T[aligned_mask, np.newaxis],
                                     hp1[:2].T[aligned_mask, np.newaxis], K0, 0, rvec, T01, useExtrinsicGuess=True,
                                     flags=cv2.SOLVEPNP_ITERATIVE)
        R01 = cv2.Rodrigues(rvec)[0].T
        T01_c = -R01.dot(T01)[:, 0]

        if not (T01_c is None or np.isinf(np.linalg.norm(T01_c))):
            reg_flow_PP = R01.T.dot(reg_flow_P) - R01.T.dot(T01_c)[:, np.newaxis]
            hpp1 = K0.dot(reg_flow_PP)
            hpp1 = hpp1 / hpp1[-1:]
            flow[bg_mask] = (hpp1 - hp0).T.reshape(h, w, 3)[bg_mask][:, :2]
            disp_change[bg_mask] = bl * K0[0, 0] / reg_flow_PP[-1].reshape(h, w)[bg_mask]

    return flow, disp_change
