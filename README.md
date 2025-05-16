# 基于面向对象的重构方案

## 流程中主要用到的类


+ VideoProcessor：用来处理视频、提取视频帧，或者将一个或者多个视频读取到Tensor中

+ SamplesBuilder：创建用于送入LLM的samples组

+ XXXHandler：用于封装不同的LLM，实现其query方法来进行查询。假如我们要利用一些模型提取特征，那就实现一个extract方法

+ Evaluator：放置所有用来评估模型的方法。写一个函数，评估某个实验产生的结果文件夹下的所有实验结果。

+ XXXPipeline：包含一整个完整的实验流程，可能需要设计一个Pipeline基类，里面放上一些需要复用的方法，或者复用流程。Pipeline需要一个名字，每次实验结果保存在以名字+次数的子文件夹内



