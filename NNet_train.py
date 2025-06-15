import os
import sys
import time
import io  # 用于内存I/O
import posixpath  # 用于处理云端对象路径

import numpy as np
from tqdm import tqdm

from utils import *

import torch
import torch.optim as optim

from NNet_structure import GameNNet, GameForNNet
from MCTS import MCTS
from torch.utils.tensorboard import SummaryWriter

args = dotdict({
    'lr': 0.001,
    'dropout': 0.3,
    'epochs': 10,
    'batch_size': 64,
    'cuda': torch.cuda.is_available(),
    'num_channels': 512,
})


class NNetWrapper():
    def __init__(self, game):
        self.nnet = GameNNet(game, args)
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()

        log_dir = os.path.join("runs", "nnet_train")
        self.writer = SummaryWriter(log_dir=log_dir)
        self.train_step = 0

        if args.cuda:
            self.nnet.cuda()

    def train(self, examples):
        """
        examples: list of (board, pi, v)
        """
        optimizer = optim.Adam(self.nnet.parameters(), lr=args.lr)

        for epoch in range(args.epochs):
            print('EPOCH ::: ' + str(epoch + 1))
            self.nnet.train()
            pi_losses = AverageMeter()
            v_losses = AverageMeter()
            batch_count = int(len(examples) / args.batch_size)

            t = tqdm(range(batch_count), desc='Training Net')
            for _ in t:
                sample_ids = np.random.randint(len(examples), size=args.batch_size)
                boards, pis, vs = list(zip(*[examples[i] for i in sample_ids]))
                boards = torch.FloatTensor(np.array(boards)).permute(0, 3, 1, 2)
                target_pis = torch.FloatTensor(np.array(pis))
                target_vs = torch.FloatTensor(np.array(vs))

                if args.cuda:
                    boards, target_pis, target_vs = boards.contiguous().cuda(), target_pis.contiguous().cuda(), target_vs.contiguous().cuda()

                out_pi, out_v = self.nnet(boards)
                l_pi = self.loss_pi(target_pis, out_pi)
                l_v = self.loss_v(target_vs, out_v)
                total_loss = l_pi + l_v

                if torch.isnan(total_loss) or torch.isinf(total_loss):
                    print("Loss is NaN or Inf! Skipping this batch.")
                    continue

                pi_losses.update(l_pi.item(), boards.size(0))
                v_losses.update(l_v.item(), boards.size(0))
                t.set_postfix(Loss_pi=pi_losses.avg, Loss_v=v_losses.avg)
                self.writer.add_scalar("Loss/Total", total_loss.item(), self.train_step)
                self.writer.add_scalar("Loss/Policy", l_pi.item(), self.train_step)
                self.writer.add_scalar("Loss/Value", l_v.item(), self.train_step)
                self.train_step += 1

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

    def predict(self, board):
        """
        board: np array of shape [H, W, 4]
        """
        board = torch.FloatTensor(board.astype(np.float32)).permute(2, 0, 1).unsqueeze(0)
        if args.cuda:
            board = board.contiguous().cuda()

        self.nnet.eval()
        with torch.no_grad():
            pi, v = self.nnet(board)

        return torch.exp(pi).data.cpu().numpy()[0], v.data.cpu().numpy()[0][0]

    def loss_pi(self, targets, outputs):
        return -torch.sum(targets * outputs) / targets.size()[0]

    def loss_v(self, targets, outputs):
        return torch.sum((targets - outputs.view(-1)) ** 2) / targets.size()[0]

    # --- 修改: 重写save_checkpoint以使用OSS ---
    def save_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar', bucket=None):
        if not bucket:
            print("OSS bucket not provided. Falling back to local save.")
            filepath = os.path.join(folder, filename)
            if not os.path.exists(folder):
                os.mkdir(folder)
            torch.save({'state_dict': self.nnet.state_dict()}, filepath)
            return

        object_key = posixpath.join(folder, filename)
        try:
            # 使用BytesIO作为内存缓冲区
            with io.BytesIO() as buffer:
                torch.save({'state_dict': self.nnet.state_dict()}, buffer)
                buffer.seek(0)  # 重置指针到缓冲区开头
                # 从内存上传到OSS
                bucket.put_object(object_key, buffer.read())
            print(f"Checkpoint saved to OSS: oss://{bucket.bucket_name}/{object_key}")
        except Exception as e:
            print(f"Failed to save checkpoint to OSS. Error: {e}")

    # --- 修改: 重写load_checkpoint以使用OSS ---
    def load_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar', bucket=None):
        if not bucket:
            print("OSS bucket not provided. Falling back to local load.")
            filepath = os.path.join(folder, filename)
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"No model in local path {filepath}")
            map_location = None if args.cuda else 'cpu'
            checkpoint = torch.load(filepath, map_location=map_location)
            self.nnet.load_state_dict(checkpoint['state_dict'])
            return

        object_key = posixpath.join(folder, filename)
        try:
            if not bucket.object_exists(object_key):
                raise FileNotFoundError(f"No model in OSS path: oss://{bucket.bucket_name}/{object_key}")

            # 从OSS下载模型到内存
            oss_object = bucket.get_object(object_key)

            # 使用BytesIO从内存中加载模型
            with io.BytesIO(oss_object.read()) as buffer:
                map_location = None if args.cuda else 'cpu'
                checkpoint = torch.load(buffer, map_location=map_location)

            self.nnet.load_state_dict(checkpoint['state_dict'])
            print(f"Checkpoint loaded from OSS: oss://{bucket.bucket_name}/{object_key}")
        except Exception as e:
            print(f"Failed to load checkpoint from OSS. Error: {e}")
            raise  # 重新抛出异常，以便上层可以捕获


# --- NNetWrapper2 ---
# 注意：如果你的项目实际使用此类，请对其进行与NNetWrapper中save/load_checkpoint类似的修改。
class NNetWrapper2():
    pass  # (代码省略，保持原样)


# The __main__ block is for local testing and doesn't need cloud modifications.
if __name__ == "__main__":
    pass  # (代码省略，保持原样)