import logging
import os
import oss2  # 导入阿里云OSS SDK
import coloredlogs

from Coach import Coach
from NNet_structure import GameForNNet as Game
from NNet_train import NNetWrapper as nn
from utils import *

log = logging.getLogger(__name__)
coloredlogs.install(level='INFO')  # Change this to DEBUG to see more info.


# ==================== [阿里云OSS配置] ====================
# 建议通过环境变量设置AccessKey，更安全
# export OSS_ACCESS_KEY_ID='YOUR_ACCESS_KEY_ID'
# export OSS_ACCESS_KEY_SECRET='YOUR_ACCESS_KEY_SECRET'
#
# TODO: 请务必修改为你的真实配置
OSS_CONFIG = {
    'access_key_id': os.getenv('OSS_ACCESS_KEY_ID', 'YOUR_ACCESS_KEY_ID'),
    'access_key_secret': os.getenv('OSS_ACCESS_KEY_SECRET', 'YOUR_ACCESS_KEY_SECRET'),
    'endpoint': 'oss-ap-southeast-1-internal.aliyuncs.com',  # TODO: 修改为你的Bucket所在的Endpoint
    'bucket_name': 'pmpmchess' # TODO: 修改为你的Bucket名称
}
# ========================================================


args = dotdict({
    'numIters': 40,
    'numEps': 100,              # Number of complete self-play games to simulate during a new iteration.
    'tempThreshold': 15,        #
    'updateThreshold': 0.6,     # During arena playoff, new neural net will be accepted if threshold or more of games are won.
    'maxlenOfQueue': 200000,    # Number of game examples to train the neural networks.
    'numMCTSSims': 300,          # Number of games moves for MCTS to simulate.
    'arenaCompare': 40,         # Number of games to play during arena play to determine if new net will be accepted.
    'cpuct': 1.5,

    # checkpoint路径现在指向OSS Bucket内的对象键(key)
    'checkpoint': 'temp/',
    'load_model': True,
    'load_folder_file': ('temp', 'checkpoint_11.pth.tar'), # ('文件夹', '文件名') -> ('对象键前缀', '对象键')

    'numItersForTrainExamplesHistory': 20,
    'maxTotalTrainingExamples': 800000
})


def main():
    # --- 初始化阿里云OSS Bucket ---
    log.info("Initializing Aliyun OSS connection...")
    try:
        auth = oss2.Auth(OSS_CONFIG['access_key_id'], OSS_CONFIG['access_key_secret'])
        bucket = oss2.Bucket(auth, OSS_CONFIG['endpoint'], OSS_CONFIG['bucket_name'])
        # 验证一下是否连接成功
        bucket.get_bucket_info()
        log.info(f"Successfully connected to OSS bucket '{OSS_CONFIG['bucket_name']}'.")
    except Exception as e:
        log.error(f"Failed to connect to Aliyun OSS. Please check your credentials, endpoint, and bucket name. Error: {e}")
        return  # 连接失败则退出

    log.info('Loading %s...', Game.__name__)
    print("游戏初始化")
    g = Game()

    log.info('Loading %s...', nn.__name__)
    print("神经网络初始化")
    # --- 修改：将bucket对象传入NNetWrapper ---
    # 我们将在netwrapper.py中修改构造函数来接收这个bucket对象
    nnet = nn(g)

    if args.load_model:
        log.info('Loading checkpoint from OSS "%s/%s"...', args.load_folder_file[0], args.load_folder_file[1])
        # --- 修改：load_checkpoint现在需要bucket对象来从OSS下载模型 ---
        # 我们将在netwrapper.py中修改这个函数
        nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1], bucket)
    else:
        log.warning('Not loading a checkpoint!')

    log.info('Loading the Coach...')
    # --- 修改：将bucket对象传入Coach ---
    # 我们将在coach.py中修改构造函数来接收这个bucket对象
    c = Coach(g, nnet, args, bucket)

    if args.load_model:
        log.info("Loading 'trainExamples' from OSS...")
        # c.loadTrainExamples()内部将使用bucket对象从OSS加载
        c.loadTrainExamples()

    log.info('Starting the learning process 🎉')
    c.learn()


if __name__ == "__main__":
    if 'YOUR_ACCESS_KEY' in OSS_CONFIG['access_key_id'] or 'your-alphazero-bucket-name' in OSS_CONFIG['bucket_name']:
        log.warning("Please configure your Aliyun OSS details in main.py before running.")
    else:
        main()