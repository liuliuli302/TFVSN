from config import *


class VideoProcessor:
    '''
    视频处理工具类
    1 视频帧抽取方法
    2 加载一个/多个视频到内存（以Tensor/ndarray形式返回）
    '''
    def __init__(self, config: VideoProcessorConfig):
        self.video_path = config.video_path
        self.frames = []
        self.current_frame_index = 0

    def load_video(self):
        # Load the video and extract frames
        pass

    def process_frame(self, frame):
        # Process the frame (e.g., apply filters, transformations)
        pass

    def save_frame(self, frame, output_path):
        # Save the processed frame to the specified output path
        pass

    def run(self):
        self.load_video()
        for frame in self.frames:
            processed_frame = self.process_frame(frame)
            self.save_frame(processed_frame, f"output/frame_{self.current_frame_index}.jpg")
            self.current_frame_index += 1
