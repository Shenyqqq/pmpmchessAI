import os
import numpy as np
from MCTS import MCTS
#from Arena import Arena
from NNet_structure import GameForNNet as Game
from NNet_train import NNetWrapper
from utils import dotdict

import logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)
from tqdm import tqdm

class Arena():
    """
    An Arena class where any 2 agents can be pit against each other.
    """

    def __init__(self, player1, player2, game, display=None):
        """
        Input:
            player 1,2: two functions that takes board as input, return action
            game: Game object
            display: a function that takes board as input and prints it (e.g.
                     display in othello/OthelloGame). Is necessary for verbose
                     mode.

        see othello/OthelloPlayers.py for an example. See pit.py for pitting
        human players/other baselines with each other.
        """
        self.player1 = player1
        self.player2 = player2
        self.game = game
        self.display = display

    def playGame(self, verbose=False):
        """
        Executes one episode of a game.

        Returns:
            either
                winner: player who won the game (1 if player1, -1 if player2)
            or
                draw result returned from the game that is neither 1, -1, nor 0.
        """
        players = [self.player2, None, self.player1]
        curPlayer = 1
        board = self.game.getInitBoard()
        it = 0

        for player in players[0], players[2]:
            if hasattr(player, "startGame"):
                player.startGame()

        while self.game.getGameEnded(board, curPlayer) == 0:
            it += 1
            if verbose:
                print("Turn ", str(it), "Player ", str(curPlayer))
                print("Board\n", self.game.game.board)
                #assert self.display
                #print("Turn ", str(it), "Player ", str(curPlayer))
                #self.display(board)
            action = players[curPlayer + 1](self.game.getCanonicalForm(board, curPlayer))

            valids = self.game.getValidMoves(self.game.getCanonicalForm(board, curPlayer), 1)

            if valids[action] == 0:
                log.error(f'Action {action} is not valid!')
                log.debug(f'valids = {valids}')
                assert valids[action] > 0

            # Notifying the opponent for the move
            opponent = players[-curPlayer + 1]
            if hasattr(opponent, "notify"):
                opponent.notify(board, action)

            board, curPlayer = self.game.getNextState(board, curPlayer, action)

        for player in players[0], players[2]:
            if hasattr(player, "endGame"):
                player.endGame()

        if verbose:
            print("Game over: Turn ", str(it), "Result ", str(self.game.getGameEnded(board, 1)))
            #assert self.display
            #print("Game over: Turn ", str(it), "Result ", str(self.game.getGameEnded(board, 1)))
            #self.display(board)
        return curPlayer * self.game.getGameEnded(board, curPlayer)

    def playGames(self, num, verbose=False):
        """
        Plays num games in which player1 starts num/2 games and player2 starts
        num/2 games.

        Returns:
            oneWon: games won by player1 (total)
            twoWon: games won by player2 (total)
            draws:  games won by nobody (total)
            p1_first_wins: games won by player1 when starting first
            p2_first_wins: games won by player2 when starting first
            p1_second_wins: games won by player1 when starting second
            p2_second_wins: games won by player2 when starting second
            draws_first_half: draws when player1 starts
            draws_second_half: draws when player2 starts
        """

        num = int(num / 2)
        oneWon = 0
        twoWon = 0
        draws = 0

        p1_first_wins = 0
        p2_second_wins = 0 # This is player2 winning when player1 started (player2 went second)
        draws_first_half = 0

        # Player 1 starts
        for _ in tqdm(range(num), desc="Arena.playGames (Player 1 Starts)"):
            gameResult = self.playGame(verbose=verbose)
            if gameResult >= 0.5:
                oneWon += 1
                p1_first_wins += 1
            elif gameResult <= -0.5:
                twoWon += 1
                p2_second_wins += 1
            else:
                draws += 1
                draws_first_half += 1

        # Swap players for the second half of games
        self.player1, self.player2 = self.player2, self.player1

        p2_first_wins = 0
        p1_second_wins = 0 # This is player1 winning when player2 started (player1 went second)
        draws_second_half = 0

        # Player 2 starts
        for _ in tqdm(range(num), desc="Arena.playGames (Player 2 Starts)"):
            gameResult = self.playGame(verbose=verbose)
            # Note: gameResult here is from the perspective of the *current* player1 (which is original player2)
            # So if gameResult is >= 0.5, it means the current player1 (original player2) won.
            if gameResult >= 0.5:
                twoWon += 1  # Original player2 wins
                p2_first_wins += 1
            elif gameResult <= -0.5:
                oneWon += 1  # Original player1 wins
                p1_second_wins += 1
            else:
                draws += 1
                draws_second_half += 1

        return {
            "oneWon_total": oneWon,
            "twoWon_total": twoWon,
            "draws_total": draws,
            "p1_first_wins": p1_first_wins,
            "p2_first_wins": p2_first_wins,
            "p1_second_wins": p1_second_wins,
            "p2_second_wins": p2_second_wins,
            "draws_when_p1_starts": draws_first_half,
            "draws_when_p2_starts": draws_second_half,
        }

# ========================
# 配置参数
# ========================
args = dotdict({
    'numMCTSSims': 300,
    'cpuct': 1,
    'arenaCompare': 40,  # 对弈局数
    'checkpoint': 'models',
'dirichlet_alpha': 0.3,
     'dirichlet_epsilon': 0,
})

# ========================
# 初始化游戏和模型
# ========================
game = Game()
nnet_current = NNetWrapper(game)
nnet_initial = NNetWrapper(game)

# 加载训练好的当前模型和初始模型
nnet_current.load_checkpoint(folder=args.checkpoint, filename='checkpoint_2.pth.tar')
#nnet_initial.load_checkpoint(folder=args.checkpoint, filename='checkpoint_4.pth.tar')

# 构建对应的 MCTS 搜索器
mcts_current = MCTS(game, nnet_current, args)
mcts_initial = MCTS(game, nnet_initial, args)

# ========================
# 对弈并统计胜负
# ========================
arena = Arena(
    lambda x: np.argmax(mcts_initial.getActionProb(x, temp=0)),
    lambda x: np.argmax(mcts_current.getActionProb(x, temp=0)),
    game
)

log.info("Starting matches: Initial Model (P1) vs Current Model (P2)")
results = arena.playGames(args.arenaCompare)

# ========================
# 结果输出
# ========================
print(results)


"""
total = init_wins + curr_wins + draws
log.info(f"Initial Wins : {init_wins}")
log.info(f"Current Wins : {curr_wins}")
log.info(f"Draws        : {draws}")
log.info(f"Win Rate (Current) : {curr_wins / total:.2%}")
"""
