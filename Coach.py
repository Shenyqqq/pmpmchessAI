import logging
import os
import sys
from collections import deque
from pickle import Pickler, Unpickler
from random import shuffle

import numpy as np
from tqdm import tqdm

from Arena import Arena
from MCTS import MCTS
from NNet_structure import GameForNNet as Game
from NNet_train import NNetWrapper
from utils import *
from torch.utils.tensorboard import SummaryWriter

log = logging.getLogger(__name__)


class Coach():
    """
    This class executes the self-play + learning. It uses the functions defined
    in Game and NeuralNet. args are specified in main.py.
    """

    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        self.pnet = self.nnet.__class__(self.game)  # the competitor network
        self.args = args
        self.mcts = MCTS(self.game, self.nnet, self.args)
        self.trainExamplesHistory = []  # history of examples from args.numItersForTrainExamplesHistory latest iterations
        self.skipFirstSelfPlay = False  # can be overriden in loadTrainExamples()
        log_dir = os.path.join("runs", "coach_learn")
        self.writer = SummaryWriter(log_dir=log_dir)
        self.learn_iteration = 0
        self.total_training_examples = 0

    def executeEpisode(self):
        """
        This function executes one episode of self-play, starting with player 1.
        As the game is played, each turn is added as a training example to
        trainExamples. The game is played till the game ends. After the game
        ends, the outcome of the game is used to assign values to each example
        in trainExamples.

        It uses a temp=1 if episodeStep < tempThreshold, and thereafter
        uses temp=0.

        Returns:
            trainExamples: a list of examples of the form (canonicalBoard, currPlayer, pi,v)
                           pi is the MCTS informed policy vector, v is +1 if
                           the player eventually won the game, else -1.
        """
        trainExamples = []
        board = self.game.getInitBoard()
        self.curPlayer = 1
        episodeStep = 0

        while True:
            episodeStep += 1
            canonicalBoard = self.game.getCanonicalForm(board, self.curPlayer)
            temp = int(episodeStep < self.args.tempThreshold)
            tempboard = np.zeros((9, 9), dtype=np.int8)
            tempboard[canonicalBoard[:, :, 0] == 1] = 1
            tempboard[canonicalBoard[:, :, 1] == 1] = -1
            #print(f"Episode: {episodeStep}. Board: \n{tempboard}")

            pi = self.mcts.getActionProb(canonicalBoard, temp=temp)
            sym = self.game.getSymmetries(canonicalBoard, pi)
            for b, p in sym:
                trainExamples.append([b, self.curPlayer, p, None])
            #print(f"Step {episodeStep}: added {len(sym)} samples, total so far: {len(trainExamples)}")
            #print(f"Step {episodeStep}: Action Prob: {pi}")

            action = np.random.choice(len(pi), p=pi)
            board, self.curPlayer = self.game.getNextState(board, self.curPlayer, action)

            r = self.game.getGameEnded(board, self.curPlayer)

            if r != 0:
                return [(x[0], x[2], r * ((-1) ** (x[1] != self.curPlayer))) for x in trainExamples]

    def learn(self):
        """
        Performs numIters iterations with numEps episodes of self-play in each
        iteration. After every iteration, it retrains neural network with
        examples in trainExamples (which has a maximum length of maxlenofQueue).
        It then pits the new neural network against the old one and accepts it
        only if it wins >= updateThreshold fraction of games.
        """

        for i in range(1, self.args.numIters + 1):
            # bookkeeping
            log.info(f'Starting Iter #{i} ...')
            # examples of the iteration
            if not self.skipFirstSelfPlay or i > 1:
                iterationTrainExamples = deque([], maxlen=self.args.maxlenOfQueue)

                for _ in tqdm(range(self.args.numEps), desc="Self Play"):
                    self.mcts = MCTS(self.game, self.nnet, self.args)  # reset search tree
                    # 使用 extend 替代 +=，对 deque 来说更标准和高效
                    iterationTrainExamples.extend(self.executeEpisode())
                    # iterationTrainExamples 的长度在每次 extend 后都会受到 maxlen 的限制
                    #print(f"[Info] Iteration generated {len(iterationTrainExamples)} training samples.")

                # save the iteration examples to the history
                self.trainExamplesHistory.append(iterationTrainExamples)
                # *** 修改内容 2: 更新总样本计数器 ***
                self.total_training_examples += len(iterationTrainExamples)

            #if len(self.trainExamplesHistory) > self.args.numItersForTrainExamplesHistory:
            #    log.warning(
            #        f"Removing the oldest entry in trainExamples. len(trainExamplesHistory) = {len(self.trainExamplesHistory)}")
            #    self.trainExamplesHistory.pop(0)
            while self.total_training_examples > self.args.maxTotalTrainingExamples and len(
                    self.trainExamplesHistory) > 0:
                removed_deque = self.trainExamplesHistory.pop(0)  # 移除最旧的迭代样本
                self.total_training_examples -= len(removed_deque)  # 从总计数中减去
                log.warning(
                    f"Removed oldest iteration's examples. Current total samples: {self.total_training_examples}"
                )
            # NB! the examples were collected using the model from the previous iteration, so (i-1)
            self.saveTrainExamples(i - 1)

            # shuffle examples before training
            trainExamples = []
            for e in self.trainExamplesHistory:
                trainExamples.extend(e)
            shuffle(trainExamples)

            # training new network, keeping a copy of the old one
            self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            self.pnet.load_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            pmcts = MCTS(self.game, self.pnet, self.args)

            self.nnet.train(trainExamples)
            nmcts = MCTS(self.game, self.nnet, self.args)

            log.info('PITTING AGAINST PREVIOUS VERSION')
            arena = Arena(lambda x: np.argmax(pmcts.getActionProb(x, temp=0)),
                          lambda x: np.argmax(nmcts.getActionProb(x, temp=0)), self.game)
            pwins, nwins, draws = arena.playGames(self.args.arenaCompare)

            log.info('NEW/PREV WINS : %d / %d ; DRAWS : %d' % (nwins, pwins, draws))
            total_games = pwins + nwins + draws
            if total_games > 0:
                new_win_rate = nwins / total_games
                prev_win_rate = pwins / total_games
                draw_rate = draws / total_games

                # ✅ 记录到 TensorBoard
                self.writer.add_scalar("Arena/New_Win_Rate", new_win_rate, self.learn_iteration)
                self.writer.add_scalar("Arena/Prev_Win_Rate", prev_win_rate, self.learn_iteration)
                self.writer.add_scalar("Arena/Draw_Rate", draw_rate, self.learn_iteration)

                self.learn_iteration += 1

            if pwins + nwins == 0 or float(nwins) / (pwins + nwins) < self.args.updateThreshold:
                log.info('REJECTING NEW MODEL')
                self.nnet.load_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            else:
                log.info('ACCEPTING NEW MODEL')
                self.nnet.save_checkpoint(folder=self.args.checkpoint, filename=self.getCheckpointFile(i))
                self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='best.pth.tar')

    def getCheckpointFile(self, iteration):
        return 'checkpoint_' + str(iteration) + '.pth.tar'

    def saveTrainExamples(self, iteration):
        folder = self.args.checkpoint
        if not os.path.exists(folder):
            os.makedirs(folder)
        filename = os.path.join(folder, self.getCheckpointFile(iteration) + ".examples")
        with open(filename, "wb+") as f:
            Pickler(f).dump(self.trainExamplesHistory)
        f.closed

    def loadTrainExamples(self):
        modelFile = os.path.join(self.args.load_folder_file[0], self.args.load_folder_file[1])
        examplesFile = modelFile + ".examples"
        if not os.path.isfile(examplesFile):
            log.warning(f'File "{examplesFile}" with trainExamples not found!')
            r = input("Continue? [y|n]")
            if r != "y":
                sys.exit()
        else:
            log.info("File with trainExamples found. Loading it...")
            with open(examplesFile, "rb") as f:
                self.trainExamplesHistory = Unpickler(f).load()
            log.info('Loading done!')

            self.total_training_examples = sum(len(d) for d in self.trainExamplesHistory)
            log.info(
                f"Loaded {len(self.trainExamplesHistory)} iterations, total samples: {self.total_training_examples}")

            # 2. 在加载后立即执行清理
            self._prune_training_examples()

            # --- 修改结束 ---

            # examples based on the model were already collected (loaded)
            self.skipFirstSelfPlay = True

    def _prune_training_examples(self):
        """
        Private helper method to prune trainExamplesHistory
        to ensure total_training_examples stays within maxTotalTrainingExamples.
        """
        initial_total = self.total_training_examples
        while self.total_training_examples > self.args.maxTotalTrainingExamples and len(self.trainExamplesHistory) > 0:
            removed_deque = self.trainExamplesHistory.pop(0)  # 移除最旧的迭代样本
            self.total_training_examples -= len(removed_deque)  # 从总计数中减去
            log.warning(
                f"Pruning: Removed oldest iteration's examples. Current total samples: {self.total_training_examples}"
            )
        if initial_total > self.args.maxTotalTrainingExamples and self.total_training_examples <= self.args.maxTotalTrainingExamples:
            log.info(
                f"Successfully pruned training examples from {initial_total} to {self.total_training_examples} samples.")
    # --- 辅助方法结束 ---

if __name__=="__main__":
    game = Game()  # 你的棋类游戏类
    nnet = NNetWrapper(game)
    args = dotdict({
        'numIters': 100,
        'numEps': 100,  # Number of complete self-play games to simulate during a new iteration.
        'tempThreshold': 15,  #
        'updateThreshold': 0.6,
        # During arena playoff, new neural net will be accepted if threshold or more of games are won.
        'maxlenOfQueue': 200000,  # Number of game examples to train the neural networks.
        'numMCTSSims': 25,  # Number of games moves for MCTS to simulate.
        'arenaCompare': 40,  # Number of games to play during arena play to determine if new net will be accepted.
        'cpuct': 1,

        'checkpoint': './temp/',
        'load_model': False,
        'load_folder_file': ('/dev/models/8x100x50', 'best.pth.tar'),
        'numItersForTrainExamplesHistory': 20,
        'num_workers': 8
    })
    mcts = MCTS(game, nnet, args)
    coach = Coach(game, nnet, args)
    trainExamples = coach.executeEpisode()
