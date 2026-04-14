import pyrealsense2 as rs
import time
import numpy as np

def calibrate_time_offset(pipeline, num_samples=50):
    """
    采集同步点，计算相机传感器时间戳（微秒）与主机时间戳（纳秒）的偏移量
    返回：偏移量（主机纳秒时间 = 相机传感器时间（纳秒） + 偏移量（纳秒））
    """
    samples = []
    print(f"正在采集{num_samples}个同步点...")
    
    for _ in range(num_samples):
        host_ts_ns = time.time_ns()
        # 获取相机帧和对应的传感器时间戳（微秒）
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        
        if not color_frame:
            continue
        
        # 相机传感器时间戳（微秒，相对时间）
        camera_ts_us = color_frame.get_frame_metadata(rs.frame_metadata_value.sensor_timestamp)
        # 转换为纳秒（1微秒 = 1000纳秒）
        camera_ts_ns = camera_ts_us * 1000
        # 主机当前时间（纳秒，绝对Unix时间戳）
        
        
        samples.append((camera_ts_ns, host_ts_ns))
        time.sleep(0.01)  # 间隔10ms采集一个样本
    
    if not samples:
        raise ValueError("未采集到有效同步点")
    
    # 分离样本数据
    camera_timestamps, host_timestamps = zip(*samples)
    
    # 计算偏移量：主机时间（纳秒） = 相机时间（纳秒） + offset（纳秒）
    offsets = np.array(host_timestamps) - np.array(camera_timestamps)
    avg_offset = np.mean(offsets)  # 平均偏移量（纳秒）
    
    print(f"时间校准完成，平均偏移量：{avg_offset:.0f}纳秒（{avg_offset/1e6:.6f}毫秒）")
    return avg_offset

def camera_ts_to_host_ts(camera_ts_us, offset_ns):
    """
    将相机传感器时间戳（微秒）转换为主机绝对时间（纳秒，Unix时间戳）
    """
    # 先转换为纳秒
    camera_ts_ns = camera_ts_us * 1000
    # 加上偏移量得到主机绝对时间（纳秒）
    return camera_ts_ns + offset_ns

def main():
    # 配置相机
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)
    
    try:
        # 1. 校准时间偏移
        offset = calibrate_time_offset(pipeline)
        
        # 2. 实时获取帧并转换时间戳
        print("\n开始实时转换时间戳...（按Ctrl+C退出）")
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue
            
            # 获取相机传感器时间戳（微秒）
            camera_ts_us = color_frame.get_frame_metadata(rs.frame_metadata_value.sensor_timestamp)
            # 转换为主机绝对时间（纳秒）
            host_time_ns = camera_ts_to_host_ts(camera_ts_us, offset)
            
            # 转换为秒用于显示（保留9位小数，对应纳秒精度）
            host_time_sec = host_time_ns / 1e9
            current_host_time_sec = time.time()
            
            # 打印结果
            print(f"相机传感器时间戳：{camera_ts_us} μs -> 转换后主机时间：{host_time_sec:.9f}秒")
            print(f"当前主机时间：{current_host_time_sec:.9f}秒，误差：{abs(host_time_sec - current_host_time_sec):.9f}秒")
            print("-" * 80)
            time.sleep(1)  # 每秒打印一次
    
    except KeyboardInterrupt:
        print("\n程序退出")
    finally:
        pipeline.stop()

if __name__ == "__main__":
    main()
