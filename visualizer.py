import pygame
import sys
from typing import List, Tuple, Set

# 初始化pygame
pygame.init()

# 颜色定义
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
GRAY = (200, 200, 200)

font_path = "C:\Windows\Fonts\等线.ttf"
class GameVisualizer:
    def __init__(self, game, cell_size=60, padding=20):
        self.game = game
        self.cell_size = cell_size
        self.padding = padding
        self.radius = cell_size // 3

        self.width = game.size * cell_size + 2 * padding
        self.height = game.size * cell_size + 2 * padding

        # 创建窗口
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("连三棋")
        self.font = pygame.font.SysFont("Microsoft YaHei", 24)

    def draw_board(self):
        self.screen.fill(WHITE)

        # 绘制网格线
        for i in range(self.game.size + 1):
            # 横线
            pygame.draw.line(
                self.screen, BLACK,
                (self.padding, self.padding + i * self.cell_size),
                (self.width - self.padding, self.padding + i * self.cell_size),
                2
            )
            # 竖线
            pygame.draw.line(
                self.screen, BLACK,
                (self.padding + i * self.cell_size, self.padding),
                (self.padding + i * self.cell_size, self.height - self.padding),
                2
            )

        # 棋子和占领区域
        for i in range(self.game.size):
            for j in range(self.game.size):
                center = (
                    self.padding + j * self.cell_size + self.cell_size // 2,
                    self.padding + i * self.cell_size + self.cell_size // 2
                )

                if self.game.controlled[i, j] == 1:  # 玩家1占领
                    pygame.draw.rect(
                        self.screen, (200, 230, 200),
                        (self.padding + j * self.cell_size,
                         self.padding + i * self.cell_size,
                         self.cell_size, self.cell_size)
                    )
                elif self.game.controlled[i, j] == -1:  # 玩家2占领
                    pygame.draw.rect(
                        self.screen, (230, 200, 200),
                        (self.padding + j * self.cell_size,
                         self.padding + i * self.cell_size,
                         self.cell_size, self.cell_size)
                    )

                # 绘制棋子
                if self.game.board[i, j] == 1:  # 玩家1
                    pygame.draw.circle(self.screen, BLACK, center, self.radius)
                elif self.game.board[i, j] == -1:  # 玩家2
                    pygame.draw.circle(self.screen, WHITE, center, self.radius)
                    pygame.draw.circle(self.screen, BLACK, center, self.radius, 2)
        #回合数
        round_text = f"回合: {self.game.round}/{self.game.max_rounds}"
        round_surface = self.font.render(round_text, True, BLACK)
        self.screen.blit(round_surface, (self.width - 150, 10))
        #控制区域
        control_text = f"黑方控制: {self.game.control_black}  白方控制: {self.game.control_white}"
        control_surface = self.font.render(control_text, True, BLACK)
        self.screen.blit(control_surface, (10, 40))  # 放在当前玩家信息下方
        #当前玩家
        status_text = f"当前玩家: {'黑方' if self.game.current_player == 1 else '白方'}"
        text_surface = self.font.render(status_text, True, BLACK)
        self.screen.blit(text_surface, (10, 10))
        #游戏结束
        if self.game.game_over:
            if self.game.winner == 0:
                result_text = "游戏结束: 平局!"
            else:
                result_text = f"游戏结束: 玩家{self.game.winner}获胜!"
            result_surface = self.font.render(result_text, True, RED)
            text_rect = result_surface.get_rect(center=(self.width // 2, self.height // 2))
            self.screen.blit(result_surface, text_rect)

    def run(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:  # 左键点击
                        # 转换鼠标坐标到棋盘坐标
                        x, y = event.pos
                        j = (x - self.padding) // self.cell_size
                        i = (y - self.padding) // self.cell_size

                        if 0 <= i < self.game.size and 0 <= j < self.game.size:
                            self.game.make_move((i, j))

            self.draw_board()
            pygame.display.flip()

        pygame.quit()
        sys.exit()


