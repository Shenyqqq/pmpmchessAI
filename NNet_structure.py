from game import Game
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np






class GameForNNet():
    """NNet接口"""
    def __init__(self, n=9):
        self.n = n
        self.game = Game(size=n)

    def getInitBoard(self):
        self.game = Game(size=self.n)
        return self._get_board_state()

    def getBoardSize(self):
        return (self.n, self.n)

    def getActionSize(self):
        return self.n * self.n

    def getNextState(self, board, player, action):
        i, j = divmod(action, self.n)

        self._load_board_state(board)
        self.game.current_player = player

        self.game.make_move((i, j))

        next_board = self._get_board_state()
        return next_board, self.game.current_player


    def getValidMoves(self, board, player):
        self._load_board_state(board)
        self.game.current_player = player
        valids = [0] * self.getActionSize()
        for (i, j) in self.game.get_valid_moves():
            valids[i * self.n + j] = 1
        return np.array(valids)

    def getGameEnded(self, board, player):
        if not self.game.game_over:
            return 0.0
        player1_score = self.game.control_black
        player2_score = self.game.control_white

        score_diff_p1_perspective = player1_score - player2_score

        rows, cols = self.getBoardSize()
        board_cells_total = rows * cols

        normalized_score_diff = 0.0
        if board_cells_total > 0:
            normalized_score_diff = score_diff_p1_perspective / board_cells_total

        final_reward_for_p1 = 1e-4

        if self.game.winner == 0:
            final_reward_for_p1 = 1e-4
        elif self.game.winner == 1:
            final_reward_for_p1 = 0.5 + (normalized_score_diff * 0.5)
            final_reward_for_p1 = min(final_reward_for_p1, 1.0)
        elif self.game.winner == -1:
            final_reward_for_p1 = -0.5 + (normalized_score_diff * 0.5)
            final_reward_for_p1 = max(final_reward_for_p1, -1.0)

        return final_reward_for_p1 * player


    def getCanonicalForm(self, board, player):
        if player == 1:
            return board
        else:
            flipped = board.copy()
            flipped[:, :, 0] = (board[:, :, 1] == 1).astype(int)  # black <-> white
            flipped[:, :, 1] = (board[:, :, 0] == 1).astype(int)
            flipped[:, :, 2] = (board[:, :, 3] == 1).astype(int)  # ctrl white <-> black
            flipped[:, :, 3] = (board[:, :, 2] == 1).astype(int)
            return flipped

    def stringRepresentation(self, board):
        return board.tobytes()

    def _get_board_state(self):
        board_1 = (self.game.board == 1).astype(np.int8)
        board_2 = (self.game.board == -1).astype(np.int8)
        ctrl_1 = (self.game.controlled == 1).astype(np.int8)
        ctrl_2 = (self.game.controlled == -1).astype(np.int8)
        return np.stack([board_1, board_2, ctrl_1, ctrl_2], axis=-1)

    def _load_board_state(self, board):
        self.game.board = np.zeros((self.n, self.n), dtype=np.int8)
        self.game.board[board[:, :, 0] == 1] = 1
        self.game.board[board[:, :, 1] == 1] = -1
        self.game.controlled = np.zeros((self.n, self.n), dtype=np.int8)
        self.game.controlled[board[:, :, 2] == 1] = 1
        self.game.controlled[board[:, :, 3] == 1] = -1

    def getSymmetries(self, board, pi):
        assert len(pi) == self.n * self.n
        pi_board = np.reshape(pi, (self.n, self.n))
        symmetries = []

        for k in range(4):  # 0°, 90°, 180°, 270°
            new_b = np.rot90(board, k, axes=(0, 1))  # rotate board
            new_p = np.rot90(pi_board, k)  # rotate policy

            symmetries.append((new_b.copy(), new_p.flatten()))
            symmetries.append((np.fliplr(new_b).copy(), np.fliplr(new_p).flatten()))  # mirror horizontally

        return symmetries



class GameNNet(nn.Module):
    def __init__(self, game, args):
        super(GameNNet, self).__init__()

        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()
        self.args = args

        # 初始卷积层：将输入通道 (4) 映射到 num_channels
        self.start_conv = nn.Conv2d(4, args.num_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.start_bn = nn.BatchNorm2d(args.num_channels)

        # 堆叠残差块
        self.res_blocks = nn.ModuleList([ResBlock(args.num_channels) for _ in range(args.num_res_blocks)])

        # === Shared layers ===
        self.flat_size = args.num_channels * self.board_x * self.board_y

        # === Policy head ===
        self.policy_conv = nn.Conv2d(args.num_channels, 2, 1)  # reduce channels to 2
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * self.board_x * self.board_y, self.action_size)

        # === Value head ===
        self.value_conv = nn.Conv2d(args.num_channels, 1, 1)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(self.board_x * self.board_y, 256)
        self.value_fc2 = nn.Linear(256, 1)

    def forward(self, s):
        # Input shape: (batch_size, 4, board_x, board_y)
        # Initial convolution
        s = F.relu(self.start_bn(self.start_conv(s)))

        # Pass through residual blocks
        for block in self.res_blocks:
            s = block(s)

        # === Policy head ===
        pi = F.relu(self.policy_bn(self.policy_conv(s)))  # (batch_size, 2, board_x, board_y)
        pi = pi.view(-1, 2 * self.board_x * self.board_y)
        pi = self.policy_fc(pi)  # (batch_size, action_size)
        pi = F.log_softmax(pi, dim=1)

        # === Value head ===
        v = F.relu(self.value_bn(self.value_conv(s)))  # (batch_size, 1, board_x, board_y)
        v = v.view(-1, self.board_x * self.board_y)
        v = F.relu(self.value_fc1(v))
        v = self.value_fc2(v)
        v = torch.tanh(v)

        return pi, v


class ResBlock(nn.Module):
    def __init__(self, num_channels):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, x):
        residual = x  # 保存输入，用于跳跃连接

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)  # 激活函数通常在BN之后，跳跃连接之前

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual  # 残差连接：将输入加到输出上
        out = F.relu(out)  # 最后再进行激活
        return out