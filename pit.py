import pygame
import sys
from Arena import Arena
from MCTS import MCTS
from NNet_structure import GameForNNet as Game
from NNet_train import NNetWrapper as NNet
import numpy as np
from utils import *
from visualizer import GameVisualizer

# 颜色定义
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)


class VisualizedArena:
    def __init__(self, game, player1, player2):
        self.game = game
        self.player1 = player1
        self.player2 = player2
        self.visualizer = GameVisualizer(game.game)

    def play_game(self):
        """执行单局游戏并返回获胜者"""
        players = [self.player2, None, self.player1]  # -1: player2, 1: player1
        cur_player = 1

        while not self.game.game.game_over:
            # 更新可视化
            self.visualizer.draw_board()
            pygame.display.flip()

            # 处理事件（保持响应）
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

            # 获取当前玩家动作
            board = self.game._get_board_state()
            canonical_board = self.game.getCanonicalForm(board, cur_player)
            action = players[cur_player + 1](canonical_board)

            # 执行动作
            next_state, next_player = self.game.getNextState(board, cur_player, action)
            self.game._load_board_state(next_state)
            self.game.game.current_player = next_player

            # 检查游戏状态
            board = self.game._get_board_state()
            self.game.game_over = self.game.getGameEnded(board, cur_player) != 0
            if self.game.game_over:
                self.game.winner = self.game.getGameEnded(board, cur_player)

            cur_player = next_player

        # 显示最终结果
        self.show_final_result()
        return self.game.game.winner

    def show_final_result(self):
        """显示最终结果并等待退出"""
        self.visualizer.draw_board()
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
            pygame.display.flip()


def main():
    # 初始化游戏和AI
    game = Game()
    nnet = NNet(game)
    nnet.load_checkpoint('./temp/', 'best.pth.tar')

    # MCTS参数
    args = dotdict({
        'numMCTSSims': 50,
        'cpuct': 1.0,
        'arenaCompare': 0  # 禁用竞技场比较
    })

    # 玩家定义
    def human_player(canonical_board):
        """人类玩家输入处理"""
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:  # 左键点击
                        x, y = event.pos
                        j = (x - 20) // 60
                        i = (y - 20) // 60

                        if 0 <= i < game.game.size and 0 <= j < game.game.size:
                            action = i * game.game.size + j
                            if game.getValidMoves(canonical_board, 1)[action]:
                                return action

    def ai_player(canonical_board):
        """AI玩家决策"""
        mcts = MCTS(game, nnet, args)
        return np.argmax(mcts.getActionProb(canonical_board, temp=0))

    # 选择对战模式
    human_vs_ai = True  # 设为False可观看AI自对战

    # 创建竞技场
    arena = VisualizedArena(
        game=game,
        player1=ai_player,
        player2=human_player
    )

    # 开始游戏
    print("游戏开始！")
    winner = arena.play_game()
    print(f"游戏结束！获胜者: {winner}")


if __name__ == "__main__":
    pygame.init()
    main()