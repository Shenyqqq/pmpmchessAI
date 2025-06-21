Okay, here's a professional and engaging `README` for your "Three-in-a-Row" AI project, in English, incorporating all your details and a bit of creative flair.

-----

# ♟️ Three-in-a-Row AI

Welcome to **Three-in-a-Row AI**, an exciting new board game where strategy meets simplicity\! Inspired by the captivating mini-game from Yostar's indie title *Pom-Pom*, this game blends the strategic depth of Gomoku (Five-in-a-Row) with the dynamic capture mechanics of Reversi (Othello). Dive in and challenge our AI, powered by the cutting-edge AlphaZero framework\!

-----

## ✨ Features

  * **Novel Hybrid Gameplay:** Experience a unique blend of "three-in-a-row" objectives and "piece flipping" mechanics.
  * **AlphaZero-Powered AI:** Our AI is trained from scratch using a custom implementation of the AlphaZero reinforcement learning framework, continually learning and improving through self-play.
  * **Interactive GUI:** Engage with the AI or play against friends using a user-friendly Pygame-based graphical interface.
      * Easily swap first/second player.
      * Observe AI self-play for insights into its strategy.
      * Enjoy player-vs-player matches.
  * **Web-Based Play:** Instantly jump into a game on our Hugging Face Space, no setup required\!
      * **Play Now:** [🤗 Hugging face](https://huggingface.co/spaces/gumigumi/pmpmchess)
      * Utilizes Gradio and Matplotlib for seamless browser-based visualization.
  * **Customizable AI Training:** We provide the tools for you to train an even stronger AI locally or in the cloud (e.g., Alibaba Cloud), allowing for further research and development.

-----

## 🧠 Behind the AI: AlphaZero Reinforcement Learning

This project's AI is built upon a modified version of the renowned **AlphaZero** framework, drawing inspiration from [suragnair/alpha-zero-general](https://github.com/suragnair/alpha-zero-general). Our implementation specifically aligns more closely with the original AlphaZero paper's training methodology.

The AI's intelligence stems from:

  * **Monte Carlo Tree Search (MCTS):** A powerful search algorithm that explores possible moves and evaluates positions by simulating thousands of games.
  * **Neural Networks:** A deep neural network is integrated with MCTS to guide the search, predict move probabilities (policy), and evaluate game states (value).
  * **Self-Play Reinforcement Learning:** The AI learns by playing millions of games against itself, iteratively refining its neural network parameters to discover optimal strategies from pure randomness.

Due to the inherent computational intensity of self-play learning and the game's complexity, our current AI's strength is limited by available training time. However, the provided codebase empowers you to continue its evolution\!

-----

## 🚀 Get Started

This project is primarily written in **Python** and leverages the **PyTorch** deep learning framework.

### Prerequisites

  * Python 3.x
  * A graphics card with CUDA support is **highly recommended** for faster AI training (install `cudatoolkit` for PyTorch).

### Installation Steps

1.  **Clone the Repository:**

    ```bash
    git clone https://github.com/your-username/your-three-in-a-row-ai.git # Replace with your actual repo
    cd your-three-in-a-row-ai
    ```

2.  **Create a Virtual Environment (Recommended):**

    ```bash
    python -m venv venv
    # Activate on macOS/Linux
    source venv/bin/activate
    # Activate on Windows
    .\venv\Scripts\activate
    ```

3.  **Install Dependencies:**

    ```bash
    pip install numpy pygame torch tqdm
    # For CUDA support (if you have an NVIDIA GPU):
    # pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 # Replace cu118 with your CUDA version
    ```

### Running the Project

Once installed, you can interact with the game and AI in several ways:

1.  **Play the Game Directly (Player vs. Player or AI vs. AI):**

    ```bash
    python game.py
    ```

2.  **Challenge the AI (Player vs. AI):**

    ```bash
    python pit.py
    ```

3.  **Train Your Own AI:**

    ```bash
    python main.py
    ```

    *(Refer to `config.py` for training parameters. We provide two versions for local and Alibaba Cloud training.)*

-----

## 📈 AI Training Parameters

Our currently deployed AI model was trained with the following parameters:

  * `numMCTSSims`: 300 (Number of MCTS simulations per move)
  * `cpuct`: 3 (Exploration constant for MCTS)
  * `numEps`: 50 (Number of self-play games per training iteration)

This model was trained for **15 iterations** and achieved an approximate **80% win rate** against a random initial network.

**💡 Tip for Training Your Own AI:**
For effective training, we highly recommend setting `numMCTSSims` to **at least 200**. Values below 200 tend to yield limited or no improvement, regardless of the number of training iterations. Increasing `numMCTSSims` will significantly enhance the AI's search depth and decision-making capabilities.

-----

## 🤝 Contribution

We welcome contributions to improve this project\! Feel free to open issues for bug reports or feature requests, or submit pull requests with your enhancements.

-----

## 📄 License

This project is open-source and available under the [Your Chosen License, e.g., MIT License]. See the `LICENSE` file for details.

-----

## 👨‍💻 Author

**gumigumi** - Project Developer

  * GitHub: [https://github.com/Shenyqqq](https://github.com/Shenyqqq/) (Replace with your actual GitHub)
  * Hugging Face: [https://huggingface.co/gumigumi](https://www.google.com/search?q=https://huggingface.co/gumigumi) (Link to your Hugging Face profile)

-----
