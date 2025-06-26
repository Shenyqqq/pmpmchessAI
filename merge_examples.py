import os
import pickle
import logging

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# 请将 'F:\\PythonProject\\pmpmchess\\temp' 替换为你实际的文件夹路径
LOAD_FOLDER = 'F:\\PythonProject\\pmpmchess\\temp'

# 这是你要合并的两个文件名
FILE_TO_MERGE_1 = 'checkpoint_4.pth.tar.examples'
FILE_TO_MERGE_2 = 'checkpoint_8.pth.tar.examples'

MERGED_FILENAME = 'merged_checkpoint.pth.tar.examples'

def load_examples(folder, filename):
    """加载训练样本文件。"""
    filepath = os.path.join(folder, filename)
    if not os.path.isfile(filepath):
        log.error(f'文件 "{filepath}" 不存在！')
        return None
    try:
        with open(filepath, "rb") as f:
            examples = pickle.Unpickler(f).load()
        log.info(f'成功加载文件: "{filename}"')
        return examples
    except Exception as e:
        log.error(f'加载文件 "{filepath}" 时发生错误: {e}')
        return None

def save_examples(folder, filename, data):
    """保存训练样本到文件。"""
    filepath = os.path.join(folder, filename)
    try:
        with open(filepath, "wb") as f:
            pickle.dump(data, f)
        log.info(f'成功保存合并后的文件到: "{filepath}"')
        return True
    except Exception as e:
        log.error(f'保存文件 "{filepath}" 时发生错误: {e}')
        return False

if __name__ == "__main__":
    log.info("--- 开始合并训练样本 ---")

    # 1. 加载第一个文件
    examples1 = load_examples(LOAD_FOLDER, FILE_TO_MERGE_1)
    if examples1 is None:
        log.error(f"无法加载第一个文件，合并操作中止。")
        exit() # 退出程序

    # 2. 加载第二个文件
    examples2 = load_examples(LOAD_FOLDER, FILE_TO_MERGE_2)
    if examples2 is None:
        log.error(f"无法加载第二个文件，合并操作中止。")
        exit() # 退出程序

    # 3. 合并数据
    # 训练样本通常是一个列表的列表 (例如，每个epoch是一个列表，包含多个 (board, pi, v) 元组)
    # 这里的合并操作是简单地将两个列表拼接起来
    merged_examples = examples1 + examples2
    log.info(f"已成功合并 {len(examples1)} 个初始样本和 {len(examples2)} 个新样本。")
    log.info(f"合并后总共有 {len(merged_examples)} 个样本。")

    # 4. 保存合并后的数据到新文件
    save_examples(LOAD_FOLDER, MERGED_FILENAME, merged_examples)

    log.info("--- 训练样本合并完成 ---")
