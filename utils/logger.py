import json
import torch
import time
from datetime import datetime
from torchinfo import summary  # 需要安装 torchinfo 包

class Logger(object):
    def __init__(self, hp):
        print("HyperParameters:")
        print(json.dumps(hp, indent=2))
        print()

        # PyTorch 版本信息
        print("PyTorch version: {}".format(torch.__version__))
        print("CUDA available: {}".format(torch.cuda.is_available()))
        if torch.cuda.is_available():
            print(f"GPU device: {torch.cuda.get_device_name(0)}")
            print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory/1024**3:.2f} GB")

        self.start_time = time.time()
        self.prev_time = self.start_time
        self.frequency = hp["log_frequency"]
        self.error_fn = None

    def get_epoch_duration(self):
        now = time.time()
        duration = datetime.fromtimestamp(now - self.prev_time).strftime("%S.%f")[:-5]
        self.prev_time = now
        return duration

    def get_elapsed(self):
        return datetime.fromtimestamp(time.time() - self.start_time).strftime("%M:%S")

    def get_error_u(self):
        return self.error_fn() if self.error_fn else 0.0

    def set_error_fn(self, error_fn):
        self.error_fn = error_fn

    def log_train_start(self, model, model_description=False):
        print("\nTraining started")
        print("================")
        if model_description:
            # 使用 torchinfo 显示模型结构
            summary(model, input_size=(1, 2), 
                   col_names=["input_size", "output_size", "num_params", "kernel_size"],
                   verbose=0)  # 关闭详细输出

    def log_train_epoch(self, epoch, loss, custom="", is_iter=False):
        if epoch % self.frequency == 0:
            name = 'nt_epoch' if is_iter else 'tf_epoch'
            print(f"{name} = {epoch:6d}  "
                  f"elapsed = {self.get_elapsed()} "
                  f"(+{self.get_epoch_duration()})  "
                  f"loss = {loss:.4e}  " + custom)

    @staticmethod
    def log_train_opt(name):
        print(f"-- Starting {name} optimization --")

    def log_train_end(self, epoch, custom=""):
        print("==================")
        print(f"Training finished (epoch {epoch}): "
              f"duration = {self.get_elapsed()}  "
              f"error = {self.get_error_u():.4e}  " + custom)

    @staticmethod
    def log_model_summary(model, input_size=(1, 2)):
        """独立模型结构打印方法"""
        summary(model, input_size=input_size,
               col_names=["input_size", "output_size", "num_params", "trainable"],
               verbose=1)