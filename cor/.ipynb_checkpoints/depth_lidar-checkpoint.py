import torch
import math
import numpy as np
from torch_scatter import scatter_max
import kitti_util
from PIL import Image
import calibration

def get_lidar_pc(depth, calib_info, image_size, max_high=1):
    H, W = image_size
    depth = depth[-H:, :W]
    cloud = depth_to_pcl(calib_info, depth, max_high=max_high)
    cloud = filter_cloud(cloud, image_size, calib_info)
    cloud = transform(cloud, calib_info, sparse_type='angular_min', start=2.0)
    return cloud


def depth_to_pcl(calib, disp, max_high=1.):
    """
    function to transform depth into pointclouds
    Args:
        calib: calibration of the image
        depth: Generated depth from depth model
    returns:
    lidar
    """
    #with torch.no_grad ():
        #disp[disp < 0] =0
    #baseline = 0.54
    #mask = disp > 0
    #depth = calib.fu * baseline / (disp + 1.0 - mask.float())
    depth = disp
    rows, cols = depth.shape
    c, r = torch.meshgrid(torch.arange(0., cols, device='cuda'),
                          torch.arange(0., rows, device='cuda'))
    points = torch.stack([c.t(), r.t(), depth], dim=0)
    points = points.reshape((3, -1))
    points = points.t()
    cloud = calib.img_to_lidar(points[:, 0], points[:, 1], points[:, 2])
    valid = (cloud[:, 0] >= 0) & (cloud[:, 2] < max_high)
    lidar = cloud[valid]

    # pad 1 in the intensity dimension
    lidar = torch.cat(
        [lidar, torch.ones((lidar.shape[0], 1), device='cuda')], 1)
    lidar = lidar.float()
    return lidar


def filter_cloud(velo_points, image_size, calib):
    H, W = image_size
    _, _, valid_inds_fov = get_lidar_in_image_fov(
        velo_points[:, :3], calib, 0, 0, W, H, True)
    velo_points = velo_points[valid_inds_fov]

    # depth, width, height
    valid_inds = (velo_points[:, 0] < 120) & \
                 (velo_points[:, 0] >= 0) & \
                 (velo_points[:, 1] < 50) & \
                 (velo_points[:, 1] >= -50) & \
                 (velo_points[:, 2] < 1.5) & \
                 (velo_points[:, 2] >= -2.5)
    velo_points = velo_points[valid_inds]
    return velo_points


def transform(points, calib_info, sparse_type, start=2.):
    if sparse_type == 'angular':
        points = random_sparse_angular(points)
    if sparse_type == 'angular_min':
        points = nearest_sparse_angular(points, start)
    if sparse_type == 'angular_numpy':
        points = points.cpu().numpy()
        points = pto_ang_map(points).astype(np.float32)
        points = torch.from_numpy(points).cuda()

    return points


def get_lidar_in_image_fov(pc_velo, calib, xmin, ymin, xmax, ymax,
                           return_more=False, clip_distance=2.0):
    ''' Filter lidar points, keep those in image FOV '''
    pts_2d, pts_rect_depth = calib.lidar_to_img(pc_velo)
    fov_inds = (pts_2d[:, 0] < xmax) & (pts_2d[:, 0] >= xmin) & \
               (pts_2d[:, 1] < ymax) & (pts_2d[:, 1] >= ymin)
    fov_inds = fov_inds & (pc_velo[:, 0] > clip_distance)
    imgfov_pc_velo = pc_velo[fov_inds, :]
    if return_more:
        return imgfov_pc_velo, pts_2d, fov_inds
    else:
        return imgfov_pc_velo


def gen_ang_map(velo_points, start=2., H=64, W=512, device='cuda'):
    dtheta = math.radians(0.4 * 64.0 / H)
    dphi = math.radians(90.0 / W)

    x, y, z, i = velo_points[:, 0], velo_points[:,
                                    1], velo_points[:, 2], velo_points[:, 3]

    d = torch.sqrt(x ** 2 + y ** 2 + z ** 2)
    r = torch.sqrt(x ** 2 + y ** 2)
    d[d == 0] = 0.000001
    r[r == 0] = 0.000001
    phi = math.radians(45.) - torch.asin(y / r)
    phi_ = (phi / dphi).long()
    phi_ = torch.clamp(phi_, 0, W - 1)

    theta = math.radians(start) - torch.asin(z / d)
    theta_ = (theta / dtheta).long()
    theta_ = torch.clamp(theta_, 0, H - 1)
    return [theta_, phi_]


def min_dist_subsample(velo_points, theta_, phi_, H, W, device='cuda'):
    N = velo_points.shape[0]

    idx = theta_ * W + phi_  # phi_ in range [0, W-1]
    depth = torch.arange(0, N, device='cuda')

    sampled_depth, argmin = scatter_max(depth, idx)
    mask = argmin[argmin != -1]
    return velo_points[mask]


def nearest_sparse_angular(velo_points, start=2., H=64, W=512, slice=1, device='cuda'):
    """
    :param H: the row num of depth map, could be 64(default), 32, 16
    :param W: the col num of depth map
    :param slice: output every slice lines
    """

    with torch.no_grad():
        theta_, phi_ = gen_ang_map(velo_points, start, H, W, device=device)

    depth_map = - torch.ones((H, W, 4), device=device)
    depth_map = min_dist_subsample(velo_points, theta_, phi_, H, W, device='cuda')
    # depth_map = depth_map[0::slice, :, :]
    depth_map = depth_map.reshape((-1, 4))
    sparse_points = depth_map[depth_map[:, 0] != -1.0]
    return sparse_points


def random_sparse_angular(velo_points, H=64, W=512, slice=1, device='cuda'):
    """
    :param velo_points: Pointcloud of size [N, 4]
    :param H: the row num of depth map, could be 64(default), 32, 16
    :param W: the col num of depth map
    :param slice: output every slice lines
    """

    with torch.no_grad():
        theta_, phi_ = gen_ang_map(velo_points, H=64, W=512, device=device)

    depth_map = - torch.ones((H, W, 4), device=device)

    depth_map = depth_map
    velo_points = velo_points
    x, y, z, i = velo_points[:, 0], velo_points[:, 1], velo_points[:, 2], velo_points[:, 3]
    theta_, phi_ = theta_, phi_

    # Currently, does random subsample (maybe keep the points with min distance)
    depth_map[theta_, phi_, 0] = x
    depth_map[theta_, phi_, 1] = y
    depth_map[theta_, phi_, 2] = z
    depth_map[theta_, phi_, 3] = i
    depth_map = depth_map.cuda()

    depth_map = depth_map[0:: slice, :, :]
    depth_map = depth_map.reshape((-1, 4))
    return depth_map[depth_map[:, 0] != -1.0]


def pto_ang_map(velo_points, H=64, W=512, slice=1):
    """
    :param H: the row num of depth map, could be 64(default), 32, 16
    :param W: the col num of depth map
    :param slice: output every slice lines
    """

    #   np.random.shuffle(velo_points)
    dtheta = np.radians(0.4 * 3.0 / H)
    dphi = np.radians(90.0 / W)

    x, y, z, i = velo_points[:, 0], velo_points[:, 1], velo_points[:, 2], velo_points[:, 3]

    d = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    r = np.sqrt(x ** 2 + y ** 2)
    d[d == 0] = 0.000001
    r[r == 0] = 0.000001
    phi = np.radians(45.) - np.arcsin(y / r)
    phi_ = (phi / dphi).astype(int)
    phi_[phi_ < 0] = 0
    phi_[phi_ >= W] = W - 1

    theta = np.radians(2.) - np.arcsin(z / d)
    theta_ = (theta / dtheta).astype(int)
    theta_[theta_ < 0] = 0
    theta_[theta_ >= H] = H - 1

    depth_map = - np.ones((H, W, 4))
    depth_map[theta_, phi_] = velo_points

    depth_map = depth_map[0::slice, :, :]
    depth_map = depth_map.reshape((-1, 4))
    depth_map = depth_map[depth_map[:, 0] != -1.0]
    return depth_map


if __name__ == "__main__":
    import torch
    import PIL
    from PIL import Image
    depth=torch.from_numpy(np.load( "/content/content/content/depth_alki.npy")).cuda()
    depth=torch.squeeze(depth, 1)
    print("depth shape",depth.shape)
  
    calib_info=kitti_util.Calib(calibration.Calibration('/content/content/content/000003.txt'))
    image=PIL.Image.open("/content/000003.png")
    single_point_cloud=get_lidar_pc(depth, calib_info, image_size, max_high=1)
    print(single_point_cloud.shape)
    single_point_cloud.astype('float32').tofile("000003.bin")