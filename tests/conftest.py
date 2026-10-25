import torch

# テストのモデルは小さいので、多コアのサーバで全コア分のスレッドを立てると取り合いでかえって遅くなる
torch.set_num_threads(min(4, torch.get_num_threads()))
