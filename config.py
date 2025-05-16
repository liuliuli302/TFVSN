class VideoProcessorConfig:
    def __init__(
        self,
        video_path
    ):
        self.video_path = video_path
        self.output_path = "output.mp4"
        self.frame_rate = 30
        self.resolution = (1920, 1080)
        self.codec = "H264"
        self.bitrate = "4000k"
        self.audio_codec = "aac"
        self.audio_bitrate = "128k"
        self.audio_sample_rate = 44100
        self.audio_channels = 2