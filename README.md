# SmartFlow AI

SmartFlow AI is an end-to-end Python project for real-time traffic signal optimization using Deep Reinforcement Learning. It simulates a three-lane traffic intersection and trains a Deep Q-Network (DQN) agent to choose green signals that reduce total waiting time and congestion.

## Live Demo

The app is deployed on Streamlit Cloud:

```text
https://smartflow-ai.streamlit.app
```

Open the live dashboard here: [SmartFlow AI Streamlit App](https://smartflow-ai.streamlit.app)

## Project Structure

```text
smartflow-ai/
  env/
    traffic_env.py
  model/
    dqn_model.py
  train.py
  app.py
  utils.py
  requirements.txt
  README.md
```

## How It Works

The custom environment models three incoming lanes. Each lane tracks queue length and accumulated waiting time. At every step, new cars arrive from a Poisson distribution, the active green lane releases cars, and the agent receives a negative reward based on total waiting time plus total queue length.

State:

```text
[lane1_count, lane2_count, lane3_count, current_signal]
```

Actions:

```text
0 = Lane 1 green
1 = Lane 2 green
2 = Lane 3 green
3 = Extend current signal
```

Reward:

```text
reward = -(total waiting time + total queue length)
```

The DQN uses a policy network, target network, epsilon-greedy exploration, and experience replay. The target network stabilizes learning by providing slower-moving Q-value targets, while replay memory reduces correlation between training samples.

## Setup

```bash
pip install -r requirements.txt
```

## Train

```bash
python train.py
```

Optional shorter test run:

```bash
python train.py --episodes 20 --max-steps 100
```

The trained model is saved to:

```text
model/dqn_traffic.pth
```

Training metrics are saved to:

```text
model/training_log.csv
```

## Run Dashboard

Live app:

```text
https://smartflow-ai.streamlit.app
```

Run locally:

```bash
streamlit run app.py
```

The Streamlit app loads the trained DQN model, simulates the intersection, and displays:

- Current queue length per lane
- Active green signal
- Waiting-time trend
- Reward trend
- Live queue visualization

## Requirements

- Python 3.10+
- PyTorch
- NumPy
- Streamlit
- Matplotlib
