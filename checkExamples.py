import os
import pickle
import numpy as np
import logging
import sys


log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

from game import Game  # <--- 确认这里导入的Game类路径正确

LOAD_FOLDER = 'F:\\PythonProject\\pmpmchess\\temp'

EXAMPLES_FILENAME = 'checkpoint_6.pth.tar.examples'


game = Game()


def load_all_train_examples(folder, filename):
    """
    根据提供的 loadTrainExamples 函数，加载所有训练样本历史。
    """
    examplesFile = os.path.join(folder, filename)

    if not os.path.isfile(examplesFile):
        log.error(f'File "{examplesFile}" with trainExamples not found!')
        return None
    else:
        log.info("File with trainExamples found. Loading it...")
        try:
            with open(examplesFile, "rb") as f:

                trainExamplesHistory = pickle.Unpickler(f).load()
            log.info('Loading done!')
            return trainExamplesHistory
        except Exception as e:
            log.error(f"Error loading trainExamples from '{examplesFile}': {e}")
            return None


def display_single_sample(sample_data, iter_index, sample_index_in_iter):
    """显示单个训练样本的棋盘、策略和价值。"""

    # 打印原始样本数据，以防解包仍有问题，便于调试
    # log.debug(f"Raw sample_data: {sample_data}")
    # log.debug(f"Raw sample_data length: {len(sample_data)}")

    try:
        # 根据你提供的 executeEpisode 返回值，每个样本应该是 (board, pi, v)
        board, pi, v = sample_data
    except ValueError as e:
        log.error(f"Error unpacking sample from iteration {iter_index}, sample {sample_index_in_iter}: {e}")
        log.error(f"Sample content was: {sample_data}")
        log.error(
            "This usually means the sample structure is not (board, pi, v). Please check your executeEpisode's return format or old example files.")
        return

    print(f"\n--- 迭代 {iter_index}, 样本 {sample_index_in_iter} ---")
    print("--- 棋盘状态 ---")
    game.display(board)

    print("\n--- 策略分布 (pi) ---")
    print(np.round(pi, 3))

    print("\n--- 游戏价值 (v) ---")
    print(f"价值: {v:.3f}")

if __name__ == "__main__":
    train_examples_history = load_all_train_examples(LOAD_FOLDER, EXAMPLES_FILENAME)

    if train_examples_history:
        total_loaded_samples = sum(len(d) for d in train_examples_history)
        log.info(f"成功加载 {len(train_examples_history)} 个迭代的历史样本，总计 {total_loaded_samples} 个样本。")

        # 我们可以选择查看第一个迭代的前 N 个样本
        num_samples_to_display = 100
        print(f"\n显示第一个迭代的前 {num_samples_to_display} 个样本:")

        if train_examples_history:  # Check if there's at least one iteration
            first_iteration_examples = train_examples_history[0]
            if not first_iteration_examples:
                log.info(f"迭代 0 没有样本。")
            else:
                print(f"\n--- 迭代 0 包含 {len(first_iteration_examples)} 个样本 ---")

                # Display the first few samples of the first iteration
                samples_in_this_iter = min(num_samples_to_display, len(first_iteration_examples))
                for j in range(samples_in_this_iter):
                    display_single_sample(first_iteration_examples[j], 0, j)

                if len(first_iteration_examples) > samples_in_this_iter:
                    print(f"... 迭代 0 还有 {len(first_iteration_examples) - samples_in_this_iter} 个样本未显示 ...")
        else:
            log.info("没有加载任何迭代。")


        print("\n--- 样本查看结束 ---")
    else:
        log.error("未能加载任何训练样本，请检查路径和文件。")
# --- 主程序 ---
"""
if __name__ == "__main__":
    train_examples_history = load_all_train_examples(LOAD_FOLDER, EXAMPLES_FILENAME)

    if train_examples_history:
        total_loaded_samples = sum(len(d) for d in train_examples_history)
        log.info(f"成功加载 {len(train_examples_history)} 个迭代的历史样本，总计 {total_loaded_samples} 个样本。")

        # 我们可以选择查看每个迭代中的前 N 个样本
        num_samples_per_iter_to_display = 24  # 默认每个迭代显示前3个样本
        print(f"\n显示每个迭代的前 {num_samples_per_iter_to_display} 个样本:")

        for i, iteration_examples in enumerate(train_examples_history):
            if not iteration_examples:
                log.info(f"迭代 {i} 没有样本。")
                continue

            print(f"\n--- 迭代 {i} 包含 {len(iteration_examples)} 个样本 ---")

            # 显示该迭代的前几个样本
            samples_in_this_iter = min(num_samples_per_iter_to_display, len(iteration_examples))
            for j in range(samples_in_this_iter):
                display_single_sample(iteration_examples[j], i, j)

            if len(iteration_examples) > samples_in_this_iter:
                print(f"... 迭代 {i} 还有 {len(iteration_examples) - samples_in_this_iter} 个样本未显示 ...")

        print("\n--- 样本查看结束 ---")
    else:
        log.error("未能加载任何训练样本，请检查路径和文件。")
"""