<div align="center">
<h1 align="center">
<img src="https://github.com/schneialexan/pongRL/blob/main/exlporation/gifs/episode_130.gif?raw=true" width="100" height="100"/>
<br></h1>
<h3>Pong Reinforcement Learning</h3>
<h3>FHGR- BSc Computational and Data Science - CDS-117</h3>

<p align="center">
<img src="https://img.shields.io/badge/tqdm-FFC107.svg?style&logo=tqdm&logoColor=black" alt="tqdm" />
<img src="https://img.shields.io/badge/Jupyter-F37626.svg?style&logo=Jupyter&logoColor=white" alt="Jupyter" />
<img src="https://img.shields.io/badge/Jinja-B41717.svg?style&logo=Jinja&logoColor=white" alt="Jinja" />
<img src="https://img.shields.io/badge/SciPy-8CAAE6.svg?style&logo=SciPy&logoColor=white" alt="SciPy" />

<img src="https://img.shields.io/badge/SymPy-3B5526.svg?style&logo=SymPy&logoColor=white" alt="SymPy" />
<img src="https://img.shields.io/badge/Python-3776AB.svg?style&logo=Python&logoColor=white" alt="Python" />
<img src="https://img.shields.io/badge/pandas-150458.svg?style&logo=pandas&logoColor=white" alt="pandas" />
<img src="https://img.shields.io/badge/NumPy-013243.svg?style&logo=NumPy&logoColor=white" alt="NumPy" />
</p>
<img src="https://img.shields.io/badge/github-license-5D6D7E" alt="GitHub license" />
<img src="https://img.shields.io/badge/github-last_commit-5D6D7E" alt="git-last-commit" />
<img src="https://img.shields.io/badge/github-commit_activity-5D6D7E" alt="GitHub commit activity" />
<img src="https://img.shields.io/badge/github-languages_top-5D6D7E" alt="GitHub top language" />
</div>

---

## 📖 Table of Contents
- [📖 Table of Contents](#-table-of-contents)
- [📍 Overview](#-overview)
- [📦 Features](#-features)
- [📂 Repository Structure](#-repository-structure)
- [⚙️ Modules](#modules)
- [🚀 Getting Started](#-getting-started)
    - [🔧 Installation](#-installation)
    - [🤖 Running ](#-running-)
    - [🧪 Tests](#-tests)
- [🛣 Roadmap](#-roadmap)
- [🤝 Contributing](#-contributing)
- [📄 License](#-license)
- [👏 Acknowledgments](#-acknowledgments)

---


## 📍 Overview

► Many different way to use Reinforcement learning for a game called Pong.

---

## 📦 Features

► INSERT-TEXT

---


## 📂 Repository Structure

```sh
└── /
    ├── .gitignore
    ├── MC.ipynb
    ├── exlporation/
    │   ├── agent_memory.py
    │   ├── debug.py
    │   ├── environment.py
    │   ├── main.py
    │   ├── preprocess_frame.py
    │   └── the_agent.py
    ├── openai_notebooks/
    │   ├── __init__.py
    │   ├── algos/
    │   ├── animation.gif
    │   ├── ping_pong_a2c.ipynb
    │   ├── ping_pong_ddqn.ipynb
    │   ├── ping_pong_dqn.ipynb
    │   ├── ping_pong_pg.ipynb
    │   ├── ping_pong_ppo.ipynb
    │   └── ping_pong_rainbow.ipynb
    ├── recent_weights.hdf5
    ├── requirements.txt
    └── src/
        ├── model_based/
        └── model_free/
```


---

## ⚙️ Modules

| File | Summary |
| --- | --- |
| [MC.ipynb](MC.ipynb) | MonteCarlo Testing |
| [requirements.txt](requirements.txt) | Requirements for the project to work |

<details closed><summary>exlporation</summary>

The best working RL, but also the most Resource and Time intensive.

| File | Summary |
| --- | --- |
| [agent_memory.py](agent_memory.py) | Memory, which the agent uses to store the necessary data|
| [debug.py](debug.py) | For debugging purposes, where more information is needed and replayability is highly valued|
| [environment.py](environment.py) | The Pong Environment |
| [main.py](main.py) | Starting and printing the GIFS|
| [preprocess_frame.py](preprocess_frame.py) | To make it easier for the CNN to learn |
| [the_agent.py](the_agent.py) | The Agent |

</details>


<details closed><summary>openai_notebooks</summary>

Complete Notebooks from openai themself. But its outdated.

| File | Summary |
| --- | --- |
| [ping_pong_a2c.ipynb](ping_pong_a2c.ipynb) | Actor-Critic Model |
| [ping_pong_ddqn.ipynb](ping_pong_ddqn.ipynb) | Double Deep Q Network |
| [ping_pong_dqn.ipynb](ping_pong_dqn.ipynb) | Deep Q Network |
| [ping_pong_pg.ipynb](ping_pong_pg.ipynb) | Policy Gradient Model |
| [ping_pong_ppo.ipynb](ping_pong_ppo.ipynb) | Proximal Policy Optimization |

<details closed><summary>algos</summary>

| File | Summary |
| --- | --- |

<details closed><summary>agents</summary>

| File | Summary |
| --- | --- |
| [__init__.py](__init__.py) | ► INSERT-TEXT |
| [a2c_agent.py](a2c_agent.py) | ► INSERT-TEXT |
| [ddqn_agent.py](ddqn_agent.py) | ► INSERT-TEXT |
| [dqn_agent.py](dqn_agent.py) | ► INSERT-TEXT |
| [ppo_agent.py](ppo_agent.py) | ► INSERT-TEXT |
| [reinforce_agent.py](reinforce_agent.py) | ► INSERT-TEXT |

</details>


<details closed><summary>models</summary>

| File | Summary |
| --- | --- |
| [__init__.py](__init__.py) | ► INSERT-TEXT |
| [actor_critic_cnn.py](actor_critic_cnn.py) | ► INSERT-TEXT |
| [ddqn_cnn.py](ddqn_cnn.py) | ► INSERT-TEXT |
| [dqn_cnn.py](dqn_cnn.py) | ► INSERT-TEXT |
| [dqn_linear.py](dqn_linear.py) | ► INSERT-TEXT |

</details>


<details closed><summary>preprocessing</summary>

| File | Summary |
| --- | --- |
| [__init__.py](__init__.py) | ► INSERT-TEXT |
| [stack_frame.py](stack_frame.py) | ► INSERT-TEXT |

</details>


<details closed><summary>utils</summary>

| File | Summary |
| --- | --- |
| [__init__.py](__init__.py) | ► INSERT-TEXT |
| [replay_buffer.py](replay_buffer.py) | ► INSERT-TEXT |

</details>


</details>


</details>


<details closed><summary>src</summary>


<details closed><summary>model_based</summary>

| File | Summary |
| --- | --- |
| [3.7 Dueling DQN with Pong.ipynb](3.7 Dueling DQN with Pong.ipynb) | Simple DQN implementation |
| [agent.py](agent.py) | corresponding Agent |
| [main.py](main.py) | Main |

<details closed><summary>methods</summary>

| File | Summary |
| --- | --- |
| [A3C.py](A3C.py) | Helpers |
| [DDQN.py](DDQN.py) | Helpers |
| [DQL.py](DQL.py) | Helpers |
| [ppf.py](ppf.py) | Helpers |

</details>


</details>


<details closed><summary>model_free</summary>
Q-Learning

| File | Summary |
| --- | --- |
| [agent.py](agent.py) |Agent |
| [debug.ipynb](debug.ipynb) | Debugging Purpose |
| [environment.py](environment.py) | Pong Environment|
| [main.py](main.py) | Main |
| [pong_episode.gif](pong_episode.gif) | One Pong Episode/Game |

</details>


</details>


---

## 🚀 Getting Started

***Dependencies***

Please ensure you have the following dependencies installed on your system:

`- ℹ️ Python=3.9`

`- ℹ️ pip install gymnasium`

`- ℹ️ pip install atari-py ale-py`

`- ℹ️ pip install -r requirements.txt`


### 🔧 Installation

1. Clone the  repository:
```sh
git clone git@github.com:schneialexan/pongRL.git
```

2. Change to the project directory:
```sh
cd pongRL
```

3. Install the dependencies:
```sh
pip install -r requirements.txt
```

### 🤖 Running 

```sh
cd exploration
python main.py
```

### 🧪 Tests
```sh
In the Jupyter Notebooks
```

---


## 🛣 Roadmap

> - [X] `ℹ️  Task 1: Implement Simple Q-Learning`
> - [X] `ℹ️  Task 2: Research other Peoples Work`
> - [X] `ℹ️  Task 3: Implement Deep Q Learning`
> - [ ] `ℹ️  TODO: Optimize Q Learning`


---

## 🤝 Contributing

Contributions are always welcome! Please follow these steps:
1. Fork the project repository. This creates a copy of the project on your account that you can modify without affecting the original project.
2. Clone the forked repository to your local machine using a Git client like Git or GitHub Desktop.
3. Create a new branch with a descriptive name (e.g., `new-feature-branch` or `bugfix-issue-123`).
```sh
git checkout -b new-feature-branch
```
4. Make changes to the project's codebase.
5. Commit your changes to your local branch with a clear commit message that explains the changes you've made.
```sh
git commit -m 'Implemented new feature.'
```
6. Push your changes to your forked repository on GitHub using the following command
```sh
git push origin new-feature-branch
```
7. Create a new pull request to the original project repository. In the pull request, describe the changes you've made and why they're necessary.
The project maintainers will review your changes and provide feedback or merge them into the main branch.

---

## 📄 License

This project is licensed under the `ℹ️  LICENSE-TYPE` License. See the [LICENSE-Type](LICENSE) file for additional info.

---

## 👏 Acknowledgments

`- ℹ️  Garvin Kruthof (Prof)`

[↑ Return](#Top)

---
