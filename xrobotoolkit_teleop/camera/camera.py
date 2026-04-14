import pyrealsense2 as rs
import numpy as np
import cv2
import open3d as o3d
import time


class Realsense():
    def __init__(self):
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        # self.spatial_filter = rs.spatial_filter()
        # self.spatial_filter.set_option(rs.option.holes_fill, 5)

        # 启动相机流
        self.pipeline.start(config)

        device = self.pipeline.get_active_profile().get_device()
        sensor = device.query_sensors()[1]
        sensor.set_option(rs.option.enable_auto_exposure, 1)

        self.s = device.first_roi_sensor()
        # 获取ROI测光区域
        # self.roi = self.s.get_region_of_interest()
        # # self.roi.min_x, self.roi.min_y, self.roi.max_x, self.roi.max_y = 250,30,450,250
        # self.roi.min_x, self.roi.min_y, self.roi.max_x, self.roi.max_y = 20,20,580,420 # 列索引
        # # 设置ROI测光区域
        # self.s.set_region_of_interest(self.roi)

        self.get_frame(drop=5)  # 扔掉垃圾帧，并且初始化内参数变量

    def get_frame(self, drop=0):
        # 扔掉垃圾帧
        for i in range(drop):
            frames = self.pipeline.wait_for_frames()

        frames = self.pipeline.wait_for_frames()

        # 对齐到彩图
        align = rs.align(rs.stream.color)
        frames = align.process(frames)

        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        if not depth_frame or not color_frame:
            raise Exception("无法获取帧数据")

        self.depth_intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics

        # 将深度数据转换为numpy数组
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # 膨胀补空洞
        kernel = np.ones((5, 5), np.uint8)
        depth_image = cv2.dilate(depth_image, kernel, iterations=3)

        return color_image, depth_image, depth_frame
    
    def get_frame_timestamp(self):
        frames = self.pipeline.wait_for_frames()
        host_ts_us = frames.get_timestamp() * 1_000_000  # 转换为纳秒

        # 对齐到彩图
        align = rs.align(rs.stream.color)
        frames = align.process(frames)
        color_frame = frames.get_color_frame()

        color_image = np.asanyarray(color_frame.get_data())

        return color_image, host_ts_us

    def vis_stream(self):
        while True:
            # Wait for a coherent pair of frames: depth and color
            color_image, depth_image, depth_frame = self.get_frame()
            # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
            # Stack both images horizontally
            images1 = np.hstack((color_image, depth_colormap))
            # Show images
            cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('RealSense', images1)
            key = cv2.waitKey(1)
            # Press esc or 'q' to close the image window
            if key & 0xFF == ord('q') or key == 27:
                cv2.destroyAllWindows()
                break

    def get_coordinate_from_pic(self, x, y, depth_frame):
        """
        根据画面坐标获取相机坐标系下的坐标
        """
        depth = depth_frame.get_distance(x, y)
        if depth == 0:
            raise Exception("深度获取失败")

        # 获取相机内参
        self.depth_intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics

        # 将像素点坐标和深度值转换为相机坐标系下的坐标
        camera_coordinate = rs.rs2_deproject_pixel_to_point(self.depth_intrinsics, (x, y), depth)
        print(f"像素点 ({x}, {y}) 相对于相机坐标系的坐标为: {camera_coordinate}， 物体中心深度为 {depth}")

        return camera_coordinate

    def get_point_cloud(self, depth_image, mask=None, vis_flag=1):

        # 深度单位变为m
        depth_image = depth_image / 1000
        rows, cols = depth_image.shape

        # 使用 meshgrid 生成行和列的网格坐标
        rows_grid, cols_grid = np.meshgrid(np.arange(rows), np.arange(cols))
        # 打印行和列的网格坐标
        pc = np.concatenate(
            [rows_grid.T[:, :, np.newaxis], cols_grid.T[:, :, np.newaxis], depth_image[:, :, np.newaxis]], axis=-1) # (H, W, 3)
        # 这时pc是一个三维array，最后一个维度的长度是3，其内容与索引值存在重叠，如pc[x, y]的值是[x, y, depth]

        pc = pc.reshape(-1, 3) # (H*W, 3)
        # 计算在相机坐标系下的坐标值
        fx, fy, cx, cy = self.depth_intrinsics.fx, self.depth_intrinsics.fy, self.depth_intrinsics.ppx, self.depth_intrinsics.ppy
        pc[:, 0] = (pc[:, 0] - cy) / fy * pc[:, 2]
        pc[:, 1] = (pc[:, 1] - cx) / fx * pc[:, 2]


        # 显示整幅图像的点云
        if vis_flag in [2]:
            point_cloud = o3d.geometry.PointCloud()
            point_cloud.points = o3d.utility.Vector3dVector(pc)  # 将点转换为Open3D的Vector3dVector格式
            mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=[0, 0, 0])
            o3d.visualization.draw_geometries([point_cloud, mesh_frame])

            if mask:
                # 如果有mask，则把区域外深度设为0后再显示一次点云
                masked_pc = pc.reshape(rows, cols, 3)
                masked_pc[mask != True, 2] = 0
                masked_pc = masked_pc.reshape(-1, 3)
                point_cloud = o3d.geometry.PointCloud()
                point_cloud.points = o3d.utility.Vector3dVector(masked_pc)
                o3d.visualization.draw_geometries([point_cloud])

        pc = pc.reshape(rows, cols, 3)
        pc = pc[:, :, [1, 0, 2]]  # 切换为行列索引

        return pc

    def get_flat_point_cloud_optmized_by_mask(self, depth_image, mask, vis_flag=1):
        # use mask to complete depth hole of flat area
        # 深度单位变为m
        depth_image = depth_image / 1000
        rows, cols = depth_image.shape
        # use mask to complete depth hole
        mask = mask.astype(bool)
        mask_depth = depth_image * mask

        # ============old===============
        # mask_depth[mask_depth > 0.72] = 0
        # avg_mask_depth = np.mean(mask_depth[mask_depth > 0.1])
        # ============old===============
        # ============new===============

        # 获取点云
        depth_image = depth_image * 1000
        pc = self.get_point_cloud(depth_image, vis_flag=vis_flag) # (H, W, 3)
        # 按照mask筛选点云
        pc = pc.reshape(-1, 3)  # array (H*W, 3)
        H, W = mask.shape
        mask = mask.flatten()  # array (H*W) dtype=bool

        # 删掉地面上的点云
        index = pc[:, 2] > 0.72
        mask[index] = False

        pc = pc[mask == True] # (n, 3)
        # 按照阈值筛选点云
        pc = pc[pc[:, 2] > 0.1]
        pc = pc[pc[:, 2] < 0.72]
        # 平面检测
        pc_o3d = o3d.geometry.PointCloud()  # 创建点云类
        pc_o3d.points = o3d.utility.Vector3dVector(pc)
        plane_model, inliers = pc_o3d.segment_plane(distance_threshold=0.005,  # 点到估计平面的最大距离，以便被认为是内点
                                                ransac_n=5,  # 用于估计平面的随机采样点的数量
                                                num_iterations=1000)  # 随机平面被采样和验证的次数
        # 上述函数的作用是平面检测，其返回值为平面方程（ax+by+cz+d=0）和内点的索引。Tuple[numpy.ndarray[numpy.float64[4, 1]], List[int]]
        pc_o3d = pc_o3d.select_by_index(inliers)  # 获得筛选平面点云

        pc = np.asarray(pc_o3d.points)
        avg_mask_depth = np.mean(pc[:, 2])
        # ============new===============

        depth_image[depth_image < 0.1] = avg_mask_depth
        depth_image[depth_image > 0.72] = avg_mask_depth
        print(f'avg_mask_depth {avg_mask_depth}')


        # 使用 meshgrid 生成行和列的网格坐标
        rows_grid, cols_grid = np.meshgrid(np.arange(rows), np.arange(cols))
        # 打印行和列的网格坐标
        pc = np.concatenate(
            [rows_grid.T[:, :, np.newaxis], cols_grid.T[:, :, np.newaxis], depth_image[:, :, np.newaxis]], axis=-1) # (H, W, 3)
        # 这时pc是一个三维array，最后一个维度的长度是3，其内容与索引值存在重叠，如pc[x, y]的值是[x, y, depth]

        pc = pc.reshape(-1, 3) # (H*W, 3)
        # 计算在相机坐标系下的坐标值
        fx, fy, cx, cy = self.depth_intrinsics.fx, self.depth_intrinsics.fy, self.depth_intrinsics.ppx, self.depth_intrinsics.ppy
        pc[:, 0] = (pc[:, 0] - cy) / fy * pc[:, 2]
        pc[:, 1] = (pc[:, 1] - cx) / fx * pc[:, 2]

        # 显示整幅图像的点云
        if vis_flag in [2]:
            point_cloud = o3d.geometry.PointCloud()
            point_cloud.points = o3d.utility.Vector3dVector(pc)  # 将点转换为Open3D的Vector3dVector格式
            mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=[0, 0, 0])
            o3d.visualization.draw_geometries([point_cloud, mesh_frame])

            # if mask:
            #     # 如果有mask，则把区域外深度设为0后再显示一次点云
            #     masked_pc = pc.reshape(rows, cols, 3)
            #     masked_pc[mask != True, 2] = 0
            #     masked_pc = masked_pc.reshape(-1, 3)
            #     point_cloud = o3d.geometry.PointCloud()
            #     point_cloud.points = o3d.utility.Vector3dVector(masked_pc)
            #     o3d.visualization.draw_geometries([point_cloud])

        pc = pc.reshape(rows, cols, 3)
        pc = pc[:, :, [1, 0, 2]]  # 切换为行列索引

        return pc, mask.reshape(H, W)

    def set_roi(self, x1=20, y1=20, x2=580, y2=420, vis_flag=1):

        roi = self.s.get_region_of_interest()
        print(f'before roi {roi.min_x}, {roi.min_y}, {roi.max_x}, {roi.max_y}')

        if vis_flag in [2]:
            color_image, depth_image, depth_frame = self.get_frame()
            # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
            cv2.rectangle(color_image, (roi.min_x, roi.min_y), (roi.max_x, roi.max_y), (0, 0, 255), 2)  # 红色边框，线条宽度为 2

            # Stack both images horizontally
            images1 = np.hstack((color_image, depth_colormap))
            # Show images
            cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('before setting roi', images1)

        roi.min_x, roi.min_y, roi.max_x, roi.max_y = x1, y1, x2, y2 # 列索引
        # 设置ROI测光区域
        self.s.set_region_of_interest(roi)
        time.sleep(0.2)
        roi = self.s.get_region_of_interest()
        print(f'after roi {roi.min_x}, {roi.min_y}, {roi.max_x}, {roi.max_y}')

        if vis_flag in [2]:
            color_image, depth_image, depth_frame = self.get_frame()
            # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
            cv2.rectangle(color_image, (roi.min_x, roi.min_y), (roi.max_x, roi.max_y), (0, 0, 255), 2)  # 红色边框，线条宽度为 2

            # Stack both images horizontally
            images1 = np.hstack((color_image, depth_colormap))
            # Show images
            cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('after setting roi', images1)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    def get_timestamp(self):
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            raise Exception("无法获取彩色帧数据")
        # 获取相机传感器时间戳（微秒，相对时间）
        camera_ts_ns = color_frame.get_timestamp() * 1_000_000  # 转换为纳秒
        return camera_ts_ns

if __name__ == '__main__':
    cam = Realsense()
    while True:
        # camera_timestamp = cam.get_timestamp()
        # print(f"相机时间戳：{camera_timestamp:.3f} 纳秒")
        # print('=' * 30)
        # time.sleep(1)
        cam.vis_stream()

        time.sleep(0.1)