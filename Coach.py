import logging
import os
import sys
from collections import deque
import pickle  # 使用pickle进行序列化
import io  # 用于处理字节流
import posixpath  # 用于处理云端对象路径
from random import shuffle

import numpy as np
from tqdm import tqdm

from Arena import Arena
from MCTS import MCTS
# from NNet_structure import GameForNNet as Game # Game不在Coach的__main__中使用
# from NNet_train import NNetWrapper            # NNetWrapper不在Coach的__main__中使用
from utils import *
from torch.utils.tensorboard import SummaryWriter

log = logging.getLogger(__name__)


class Coach():
    """
    This class executes the self-play + learning. It uses the functions defined
    in Game and NeuralNet. args are specified in main.py.
    """

    # --- 修改: 构造函数增加 bucket 参数 ---
    def __init__(self, game, nnet, args, bucket):
        self.game = game
        self.nnet = nnet
        self.pnet = self.nnet.__class__(self.game)  # the competitor network
        self.args = args
        self.bucket = bucket  # 保存OSS bucket对象
        self.mcts = MCTS(self.game, self.nnet, self.args)
        self.trainExamplesHistory = []  # history of examples from args.numItersForTrainExamplesHistory latest iterations
        self.skipFirstSelfPlay = False  # can be overriden in loadTrainExamples()

        # 注意: TensorBoard日志仍然保存在本地。在云服务器上，这通常没问题。
        # 训练结束后，可以将 "runs" 文件夹整体上传到OSS进行归档。
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
            decay_end_step = 25  # temp dacay in first 25 steps
            initial_temp = 1.0
            final_temp = 0
            if episodeStep <= decay_end_step:
                # Linear decay from initial temp to final temp
                decay_progress = (episodeStep - 1) / decay_end_step
                temp_value = initial_temp - (initial_temp - final_temp) * decay_progress
            else:
                temp_value = final_temp

            pi = self.mcts.getActionProb(canonicalBoard, temp=temp_value)
            sym = self.game.getSymmetries(canonicalBoard, pi)
            for b, p in sym:
                trainExamples.append([b, self.curPlayer, p, None])

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
            log.info(f'Starting Iter #{i} ...')
            if not self.skipFirstSelfPlay or i > 1:
                iterationTrainExamples = deque([], maxlen=self.args.maxlenOfQueue)
                for _ in tqdm(range(self.args.numEps), desc="Self Play"):
                    self.mcts = MCTS(self.game, self.nnet, self.args)
                    iterationTrainExamples.extend(self.executeEpisode())
                self.trainExamplesHistory.append(iterationTrainExamples)
                self.total_training_examples += len(iterationTrainExamples)

            while self.total_training_examples > self.args.maxTotalTrainingExamples and len(
                    self.trainExamplesHistory) > 0:
                removed_deque = self.trainExamplesHistory.pop(0)
                self.total_training_examples -= len(removed_deque)
                log.warning(
                    f"Removed oldest iteration's examples. Current total samples: {self.total_training_examples}")

            self.saveTrainExamples(i - 1)

            trainExamples = []
            for e in self.trainExamplesHistory:
                trainExamples.extend(e)
            shuffle(trainExamples)

            # --- 修改: 调用save/load_checkpoint时传入bucket对象 ---
            self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar', bucket=self.bucket)
            self.pnet.load_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar', bucket=self.bucket)
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
                self.writer.add_scalar("Arena/New_Win_Rate", new_win_rate, self.learn_iteration)
                self.learn_iteration += 1

            if pwins + nwins == 0 or float(nwins) / (pwins + nwins) < self.args.updateThreshold:
                log.info('REJECTING NEW MODEL')
                self.nnet.load_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar', bucket=self.bucket)
            else:
                log.info('ACCEPTING NEW MODEL')
                self.nnet.save_checkpoint(folder=self.args.checkpoint, filename=self.getCheckpointFile(i),
                                          bucket=self.bucket)
                self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='best.pth.tar', bucket=self.bucket)

    def getCheckpointFile(self, iteration):
        return 'checkpoint_' + str(iteration) + '.pth.tar'

    # --- 修改: 重写saveTrainExamples以使用OSS ---
    def saveTrainExamples(self, iteration):
        if not self.bucket:
            log.error("OSS Bucket not configured. Cannot save training examples.")
            return

        object_key = posixpath.join(self.args.checkpoint, self.getCheckpointFile(iteration) + ".examples")

        try:
            # 将训练历史数据序列化为bytes
            data_bytes = pickle.dumps(self.trainExamplesHistory)
            # 上传到OSS
            self.bucket.put_object(object_key, data_bytes)
            log.info(f"Training examples saved to OSS: oss://{self.bucket.bucket_name}/{object_key}")
        except Exception as e:
            log.error(f"Failed to save training examples to OSS object '{object_key}'. Error: {e}")

    # --- 修改: 重写loadTrainExamples以使用OSS ---
    def loadTrainExamples(self):
        if not self.bucket:
            log.error("OSS Bucket not configured. Cannot load training examples.")
            return

        model_folder, model_file = self.args.load_folder_file
        object_key = posixpath.join(model_folder, model_file) + ".examples"

        try:
            if not self.bucket.object_exists(object_key):
                log.warning(f'Training examples object not found in OSS: "{object_key}"')
                r = input("Continue without loading examples? [y|n]")
                if r != "y":
                    sys.exit()
                return

            log.info(f"Found training examples in OSS: '{object_key}'. Loading...")
            # 从OSS下载对象
            oss_object = self.bucket.get_object(object_key)
            # 读取对象内容（bytes）
            data_bytes = oss_object.read()

            # 反序列化bytes为Python对象
            self.trainExamplesHistory = pickle.loads(data_bytes)
            log.info('Loading from OSS done!')

            self.total_training_examples = sum(len(d) for d in self.trainExamplesHistory)
            log.info(
                f"Loaded {len(self.trainExamplesHistory)} iterations, total samples: {self.total_training_examples}")

            self._prune_training_examples()
            self.skipFirstSelfPlay = True

        except Exception as e:
            log.error(f"Failed to load training examples from OSS object '{object_key}'. Error: {e}")
            sys.exit()

    def _prune_training_examples(self):
        """
        Private helper method to prune trainExamplesHistory
        to ensure total_training_examples stays within maxTotalTrainingExamples.
        """
        initial_total = self.total_training_examples
        while self.total_training_examples > self.args.maxTotalTrainingExamples and len(self.trainExamplesHistory) > 0:
            removed_deque = self.trainExamplesHistory.pop(0)
            self.total_training_examples -= len(removed_deque)
            log.warning(
                f"Pruning: Removed oldest iteration's examples. Current total samples: {self.total_training_examples}")
        if initial_total > self.args.maxTotalTrainingExamples and self.total_training_examples <= self.args.maxTotalTrainingExamples:
            log.info(
                f"Successfully pruned training examples from {initial_total} to {self.total_training_examples} samples.")


# The __main__ block is for local testing/debugging and doesn't need cloud modifications.
# It will raise an error if run directly because it doesn't provide the 'bucket' object to Coach.
if __name__ == "__main__":
    from NNet_structure import GameForNNet as Game
    from NNet_train import NNetWrapper

    game = Game()
    nnet = NNetWrapper(game)
    args = dotdict({
        'numIters': 100, 'numEps': 100, 'tempThreshold': 15, 'updateThreshold': 0.6,
        'maxlenOfQueue': 200000, 'numMCTSSims': 25, 'arenaCompare': 40, 'cpuct': 1,
        'checkpoint': './temp/', 'load_model': False,
        'load_folder_file': ('/dev/models/8x100x50', 'best.pth.tar'),
        'numItersForTrainExamplesHistory': 20, 'num_workers': 8
    })
    # This will fail because the Coach now requires a 'bucket' argument.
    # coach = Coach(game, nnet, args)
    # trainExamples = coach.executeEpisode()
    print("The __main__ block in coach.py is intended for local testing and is not cloud-enabled.")
    print("Please run the project from main.py.")