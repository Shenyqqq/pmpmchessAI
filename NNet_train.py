import os
import sys
import time

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
        self.nnet = GameNNet(game, args)  # 必须是支持4通道输入的网络
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()

        log_dir = os.path.join("runs", "nnet_train")
        self.writer = SummaryWriter(log_dir=log_dir)
        self.train_step = 0  # 用于记录 batch 数

        if args.cuda:
            self.nnet.cuda()

    def train(self, examples):
        """
        examples: list of (board [H, W, 4], pi [action_size], v [float])
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
                boards = torch.FloatTensor(np.array(boards))  # [B, H, W, 4]
                boards = boards.permute(0, 3, 1, 2)  # => [B, 4, H, W]

                target_pis = torch.FloatTensor(np.array(pis))       # [B, action_size]
                target_vs = torch.FloatTensor(np.array(vs))         # [B]

                if args.cuda:
                    boards = boards.contiguous().cuda()
                    target_pis = target_pis.contiguous().cuda()
                    target_vs = target_vs.contiguous().cuda()

                # Forward
                out_pi, out_v = self.nnet(boards)  # out_pi: [B, action_size], out_v: [B, 1]
                l_pi = self.loss_pi(target_pis, out_pi)
                l_v = self.loss_v(target_vs, out_v)
                total_loss = l_pi + l_v
                if torch.isnan(total_loss) or torch.isinf(total_loss):
                    print("Loss is NaN or Inf! Skipping this batch.")
                    continue

                # record loss
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
        returns: (pi [action_size], v [float])
        """
        board = torch.FloatTensor(board.astype(np.float32))
        board = board.permute(2, 0, 1).unsqueeze(0)  # [1, 4, H, W]

        if args.cuda:
            board = board.contiguous().cuda()

        self.nnet.eval()
        with torch.no_grad():
            pi, v = self.nnet(board)

        return torch.exp(pi).data.cpu().numpy()[0], v.data.cpu().numpy()[0][0]

    def loss_pi(self, targets, outputs):
        # outputs are log probabilities
        return -torch.sum(targets * outputs) / targets.size()[0]

    def loss_v(self, targets, outputs):
        return torch.sum((targets - outputs.view(-1)) ** 2) / targets.size()[0]

    def save_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            print("Checkpoint Directory does not exist! Making directory {}".format(folder))
            os.mkdir(folder)
        else:
            print("Checkpoint Directory exists!")
        torch.save({'state_dict': self.nnet.state_dict()}, filepath)

    def load_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError("No model in path {}".format(filepath))
        map_location = None if args.cuda else 'cpu'
        checkpoint = torch.load(filepath, map_location=map_location)
        self.nnet.load_state_dict(checkpoint['state_dict'])

class NNetWrapper2():
    def __init__(self, game):
        self.nnet = GameNNet(game, args)
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()

        if args.cuda:
            self.nnet.cuda()

    def train(self, examples):
        """
        examples: list of examples, each example is of form (board, pi, v)
        """
        optimizer = optim.Adam(self.nnet.parameters())

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
                boards = torch.FloatTensor(np.array(boards).astype(np.float64))
                target_pis = torch.FloatTensor(np.array(pis))
                target_vs = torch.FloatTensor(np.array(vs).astype(np.float64))

                # predict
                if args.cuda:
                    boards, target_pis, target_vs = boards.contiguous().cuda(), target_pis.contiguous().cuda(), target_vs.contiguous().cuda()

                # compute output
                out_pi, out_v = self.nnet(boards)
                l_pi = self.loss_pi(target_pis, out_pi)
                l_v = self.loss_v(target_vs, out_v)
                total_loss = l_pi + l_v

                # record loss
                pi_losses.update(l_pi.item(), boards.size(0))
                v_losses.update(l_v.item(), boards.size(0))
                t.set_postfix(Loss_pi=pi_losses, Loss_v=v_losses)

                # compute gradient and do SGD step
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

    def predict(self, board):
        """
        board: np array with board
        """
        # timing
        start = time.time()

        # preparing input
        board = torch.FloatTensor(board.astype(np.float64))
        if args.cuda: board = board.contiguous().cuda()
        board = board.view(1, self.board_x, self.board_y)
        self.nnet.eval()
        with torch.no_grad():
            pi, v = self.nnet(board)

        # print('PREDICTION TIME TAKEN : {0:03f}'.format(time.time()-start))
        return torch.exp(pi).data.cpu().numpy()[0], v.data.cpu().numpy()[0]

    def loss_pi(self, targets, outputs):
        return -torch.sum(targets * outputs) / targets.size()[0]

    def loss_v(self, targets, outputs):
        return torch.sum((targets - outputs.view(-1)) ** 2) / targets.size()[0]

    def save_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            print("Checkpoint Directory does not exist! Making directory {}".format(folder))
            os.mkdir(folder)
        else:
            print("Checkpoint Directory exists! ")
        torch.save({
            'state_dict': self.nnet.state_dict(),
        }, filepath)

    def load_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        # https://github.com/pytorch/examples/blob/master/imagenet/main.py#L98
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise ("No model in path {}".format(filepath))
        map_location = None if args.cuda else 'cpu'
        checkpoint = torch.load(filepath, map_location=map_location)
        self.nnet.load_state_dict(checkpoint['state_dict'])


if __name__ == "__main__":

    board = GameForNNet().getInitBoard()
    print(board.shape)
    board_tensor = torch.tensor(board.transpose(2, 0, 1)).unsqueeze(0).float()  # -> (1, 4, n, n)
    print(board_tensor.shape)
    game = GameForNNet()
    nnet = GameNNet(game, args)
    pi, v = nnet(board_tensor)  # 应返回 (1, action_size), (1, 1)
    print(pi.shape,v.shape)

    from utils import *
    args2 = dotdict({
        'numIters': 1000,
        'numEps': 100,              # Number of complete self-play games to simulate during a new iteration.
        'tempThreshold': 15,        #
        'updateThreshold': 0.6,     # During arena playoff, new neural net will be accepted if threshold or more of games are won.
        'maxlenOfQueue': 200000,    # Number of game examples to train the neural networks.
        'numMCTSSims': 25,          # Number of games moves for MCTS to simulate.
        'arenaCompare': 40,         # Number of games to play during arena play to determine if new net will be accepted.
        'cpuct': 1,

        'checkpoint': './temp/',
        'load_model': False,
        'load_folder_file': ('/dev/models/8x100x50','best.pth.tar'),
        'numItersForTrainExamplesHistory': 20,

    })

    nnet = NNetWrapper(game)
    mcts = MCTS(game, nnet, args2)
    board = game.getInitBoard()
    canonical = game.getCanonicalForm(board, player=1)
    probs = mcts.getActionProb(canonical, temp=1)
    print(np.sum(probs), probs)  # sum应为1，每个元素对应一个动作
