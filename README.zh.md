[中文](https://github.com/Shenyqqq/pmpmchessAI/edit/NNet-modified/README.zh.md)   [English](https://github.com/Shenyqqq/pmpmchessAI/blob/NNet-modified/README.md)

-----

# 泡姆棋 AI

欢迎来到**泡姆棋 AI**，这是一款将策略与简洁完美结合的全新棋盘游戏！灵感来源于 Hypergryph 独立游戏《Popucom》中引人入胜的迷你游戏，本作融合了五子棋（Gomoku）的战略深度与翻转棋（Reversi/Othello）的动态棋子捕获机制。快来挑战由尖端 AlphaZero 框架驱动的 AI 吧！

-----

## 功能特性

  * **新颖的混合玩法：** 体验“三子成线”目标与“棋子翻转”机制的独特融合。
  * **AlphaZero 驱动的 AI：** AI 使用 AlphaZero 强化学习框架的自定义实现从零开始训练，通过自我对弈持续学习和改进。
  * **交互式 GUI：** 使用基于 Pygame 的用户友好图形界面与 AI 对战或与朋友对弈。
      * 轻松切换先手/后手玩家。
      * 观察 AI 自我对弈，深入了解其策略。
      * 享受玩家对战玩家的比赛。
  * **网页版游戏：** 在 Hugging Face Space 上即时开始游戏，无需任何设置！
      * **立即体验：** [🤗 Hugging Face](https://huggingface.co/spaces/gumigumi/pmpmchess)
      * 利用 Gradio 和 Matplotlib 实现无缝的浏览器内可视化。
  * **可定制的 AI 训练：** 我提供了代码，让您可以在本地或云端（例如阿里云）训练出更强大的 AI，以进行进一步的研究和开发。

-----

## 基于 AlphaZero 强化学习

本项目的 AI 基于修改版的著名 **AlphaZero** 框架，灵感来源于 [suragnair/alpha-zero-general](https://github.com/suragnair/alpha-zero-general)。我的实现更接近原始 AlphaZero 论文的训练方法。

AI 的智能来源于：

  * **蒙特卡洛树搜索（MCTS）：** 一种强大的搜索算法，通过模拟数千局游戏来探索可能的走法并评估局面。
  * **神经网络：** 深度神经网络与 MCTS 集成，用于指导搜索、预测走法概率（策略）和评估游戏状态（价值）。
  * **自我对弈强化学习：** AI 通过与自己进行数百万次对弈来学习，迭代地优化其神经网络参数，从纯粹的随机性中发现最优策略。

由于自我对弈学习固有的计算密集性以及游戏的复杂性，当前 AI 的强度受限于可用的训练时间。但是，所提供的代码库使您能够继续其演进！

-----

## 开始使用

本项目主要使用 **Python** 编写，并利用 **PyTorch** 深度学习框架。

### 先决条件

  * Python 3.x
  * **强烈推荐**带有 CUDA 支持的显卡，以加快 AI 训练（为 PyTorch 安装 `cudatoolkit`）。

### 安装步骤

1.  **克隆仓库：**

    ```cmd
    git clone https://github.com/Shenyqqq/pmpmchessAI.git 
    ```

2.  **创建虚拟环境（推荐）：**

    ```cmd
    python -m venv venv
    # 在 Windows 上激活
    .\venv\Scripts\activate
    ```

3.  **安装依赖：**

    ```cmd
    pip install numpy pygame torch tqdm
    # 对于 CUDA 支持（如果您有 NVIDIA GPU）：
    # pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 # 将 cu118 替换为您的 CUDA 版本
    ```

### 运行项目

安装完成后，您可以通过多种方式与游戏和 AI 交互：

1.  **直接玩游戏（玩家 vs 玩家）：**

    ```cmd
    python game.py
    ```

2.  **挑战 AI（玩家 vs AI 或 AI vs AI）：**

    ```cmd
    python pit.py
    ```

3.  **训练您自己的 AI：**

    ```cmd
    python main.py
    ```

    注意：（我提供了 *for-aliyun* 分支中用于阿里云训练的版本）

-----

## AI 训练参数

我们当前部署的 AI 模型使用以下参数进行训练：

  * `numMCTSSims`：300（每次走法的 MCTS 模拟次数）
  * `cpuct`：3（MCTS 的探索常数）
  * `numEps`：80（每次训练迭代的自我对弈游戏数量）
  * `dirichlet_alpha`：0.1
  * `dirichlet_epsilon`：0.25
  * `updateThreshold`：0.6（在竞技场对弈中，如果新神经网络获胜的游戏达到或超过此阈值，则会被接受）

此模型经过 **25 次迭代**训练，对抗随机模型达到了 **86.8% 的胜率**。您可以通过运行 *PitAgainstRandomModel.py* 来测试您的模型。

**💡 训练您自己的 AI 提示：**
为了有效训练，我强烈建议将 `numMCTSSims` 设置为**至少 200**。低于 200 的值往往效果有限或没有改进，无论训练迭代次数如何。增加 `numMCTSSims` 将显著增强 AI 的搜索深度和决策能力。

-----

## 贡献

欢迎为改进本项目做出贡献！欢迎提出问题（bug 报告或功能请求）或提交拉取请求以增强功能。

-----

## 许可证

暂无。

-----

## 作者

**gumigumi** - 项目开发者

  * GitHub: [https://github.com/Shenyqqq](https://github.com/Shenyqqq/)
  * Hugging Face: [https://huggingface.co/gumigumi](https://huggingface.co/gumigumi)

-----
