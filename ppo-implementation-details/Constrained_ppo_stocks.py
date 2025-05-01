import argparse
import os
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from distutils.util import strtobool
import matplotlib.pyplot as plt

from torch.utils.tensorboard import SummaryWriter
import time
import random
import numpy as np
import torch
from torch.distributions.normal import Normal
import sys
sys.path.append(r"C:\Users\derek\Documents\PPO_discrete\creating_PPO\ppo-implementation-details")
from portfolio import PortfolioEnv
import pandas as pd
from gymnasium.envs.registration import register

register(
    id="PortfolioEnv-v0",
    entry_point="portfolio:PortfolioEnv",
)
import gymnasium as gym
import os
os.environ["WANDB_DISABLE_GYM"] = "true"

import yfinance as yf
import pandas as pd

stocks = [
    "AAPL", "MSFT", "GOOG", "AMZN", "META", "TSLA", "NVDA", "JPM", "V", "PG",
    "UNH", "HD", "MA", "BAC", "PFE", "XOM", "VZ", "KO", "DIS", "ADBE",
    "NFLX", "PEP", "INTC", "CSCO", "T", "CVX", "MRK", "ABT", "CRM", "AVGO",
    "WMT", "MCD", "TMO", "COST", "TXN", "LIN", "NKE", "AMGN", "MDT", "ACN",
    "LLY", "HON", "LOW", "NEE", "SBUX", "QCOM", "BMY", "IBM", "CAT", "GE"
]

# Download
data = yf.download(stocks, start="2015-01-01", end="2024-01-01")

# Select the "Close" prices
data = data["Close"]

# Reset index
data = data.reset_index()

# Save
data.to_csv("real_stock_prices.csv", index=False)


# dates = pd.date_range(start="2000-01-01", periods=1000)
# stocks =["A", "B", "C", "D","E","F","G","H","I","J","K","L","M","N","O","P","Q",
#                      "R", "S", "T", "U","V","W","X","Y","Z","AA","BB","CC","DD","EE","FF","GG","HH",
#                      "II", "JJ", "KK", "LL","MM","NN","OO","PP","QQ","RR","SS","TT","UU","VV","WW","XX","YY",
#                      "ZZ"]
#
# price_data = {
#     "Date": dates
# }
#
# for stock in stocks:
#     start_price = np.random.uniform(50, 300)
#     daily_returns = np.random.normal(0, 0.01, size=len(dates))
#     prices = start_price * (1 + daily_returns).cumprod()
#     price_data[stock] = np.round(prices, 2)
#
# df = pd.DataFrame(price_data)
# df.to_csv("stock_pricess.csv", index=False)
price_data = pd.read_csv("real_stock_prices.csv")
price_data = price_data.drop(columns="Date")


# prices = price_data[["A", "B", "C", "D","E","F","G","H","I","J","K","L","M","N","O","P","Q",
#                      "R", "S", "T", "U","V","W","X","Y","Z","AA","BB","CC","DD","EE","FF","GG","HH",
#                      "II", "JJ", "KK", "LL","MM","NN","OO","PP","QQ","RR","SS","TT","UU","VV","WW","XX","YY",
#                      "ZZ"]].values


def layer_init(layer, std=np.sqrt(2),bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer
class Agent(nn.Module):
    def __init__(self, envs, max_cardinality, min_cardinality, min_weight=0.0, max_weight=1.0):
        super(Agent, self).__init__()
        self.max_cardinality = max_cardinality
        self.min_cardinality = min_cardinality
        self.min_weight = min_weight
        self.max_weight = max_weight

        obs_dim = np.array(envs.single_observation_space.shape).prod()
        act_dim = np.prod(envs.single_action_space.shape)

        # Shared feature extractor
        self.actor = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
        )

        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(64, act_dim), std=0.01),
        )

        self.actor_logstd = nn.Parameter(torch.zeros(1, act_dim))

        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        if torch.isnan(x).any():
            print("Input contains NaN values!")

        hidden = self.actor(x)
        mu = self.actor_mean(hidden)
        std = torch.exp(self.actor_logstd)

        # Clamp to avoid numerical issues
        mu = torch.clamp(mu, min=-10.0, max=10.0)
        std = torch.clamp(std, min=1e-3, max=2.0)

        dist = torch.distributions.Normal(mu, std)

        if action is None:
            raw_action = dist.rsample()  # rsample = differentiable sampling
            log_prob = dist.log_prob(raw_action).sum(-1)
            entropy = dist.entropy().sum(-1)

            weights = torch.softmax(raw_action, dim=-1)
            weights = self.apply_cardinality_constraints(weights)
        else:
            # Compute log_prob and entropy from the provided action
            log_prob = dist.log_prob(action).sum(-1)
            entropy = dist.entropy().sum(-1)
            weights = self.apply_cardinality_constraints(action)

        value = self.critic(x)

        return weights, log_prob, entropy, value

    def apply_cardinality_constraints(self, action):
        """
        Ensure the action respects the min and max cardinality constraints.
        Accepts a tensor of shape [batch_size, num_assets] or [num_assets].
        """
        if action.dim() == 1:
            action = action.unsqueeze(0)  # make it (1, num_assets)

        constrained = []

        for act in action:
            sorted_indices = torch.argsort(act, descending=True)
            num_assets = act.shape[0]

            # Determine which indices to keep
            nonzero_count = (act > 1e-5).sum().item()

            if nonzero_count < self.min_cardinality:
                # keep top min_cardinality assets
                selected_indices = sorted_indices[:self.min_cardinality]
            elif nonzero_count > self.max_cardinality:
                #keep only top max_cardinality assets
                selected_indices = sorted_indices[:self.max_cardinality]
            else:
                # keep only the non-zero ones
                selected_indices = (act > 1e-5).nonzero(as_tuple=True)[0]

            new_act = torch.zeros_like(act)
            new_act[selected_indices] = act[selected_indices]
            new_act = new_act / (new_act.sum() + 1e-8)  # Normalize safely

            constrained.append(new_act)

        return torch.stack(constrained) if len(constrained) > 1 else constrained[0]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-name', type=str, default=os.path.basename(__file__).rstrip(".py"))
    parser.add_argument('--gym-id', type=str, default="PortfolioEnv-v0")
    parser.add_argument('--learning-rate', type=float, default=2.5e-4)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--total-timesteps', type=int, default=100)
    parser.add_argument('--torch-deterministic', type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=True)
    parser.add_argument('--cudas', type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=True)
    parser.add_argument('--track', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True)
    parser.add_argument('--derek-project-name', type=str, default="cleanRL")
    parser.add_argument('--derek-entity', type=str, default=None)
    parser.add_argument('--capture-video', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True)
    parser.add_argument('--num-envs', type=int, default=1)
    parser.add_argument('--num-steps',type=int, default=128)
    parser.add_argument('--anneal-lr', type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=True)
    parser.add_argument('--gae',type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=True)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--gae-lambda', type=float, default=0.95)
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=True)
    parser.add_argument("--clip-coef", type=float, default=0.2)
    parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--num-minibatches", type=int, default=4)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--target-kl", type=float, default=None)
    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    return args


def make_env(gym_id, seed, idx, capture_video, run_name, price_data):
    def thunk():
        # Create the PortfolioEnv environment
        if capture_video and idx == 0:
            env = PortfolioEnv(price_data=price_data)
            # Wrapping the environment with multiple wrappers
            env = gym.wrappers.RecordEpisodeStatistics(env)
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}", episode_trigger=lambda x: True)
            env = gym.wrappers.ClipAction(env)
            env = gym.wrappers.NormalizeObservation(env)
            env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10),
                                                    observation_space=env.observation_space)
            env = gym.wrappers.NormalizeReward(env)
            env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        else:
            env = PortfolioEnv(price_data=price_data)
            # Wrapping the environment with multiple wrappers
            env = gym.wrappers.RecordEpisodeStatistics(env)
            env = gym.wrappers.ClipAction(env)
            env = gym.wrappers.NormalizeObservation(env)
            env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10),
                                                    observation_space=env.observation_space)
            env = gym.wrappers.NormalizeReward(env)
            env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))

        # Debugging: Print type of env to check if it's a valid gym.Env
        print(f"Created environment: {type(env)}")

        # Set the seed for reproducibility
        env.reset(seed=seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)

        return env

    return thunk

def evaluate(agent, env, device, episodes=5, render=False):
    agent.eval()  # Set the agent to evaluation mode
    returns = []

    for ep in range(episodes):
        obs, _ = env.reset()
        obs = torch.tensor(obs, dtype=torch.float32).to(device)
        done = False
        total_reward = 0
        step = 0

        while not done:
            with torch.no_grad():
                action, _, _, _ = agent.get_action_and_value(obs)
            obs, reward, terminated, truncated, info = env.step(action.cpu().numpy())
            # print(f"Step {step}: Stock Weights = {action.cpu().numpy()}")
            obs = torch.tensor(obs, dtype=torch.float32).to(device)
            done = terminated or truncated
            total_reward += reward
            step += 1

            if render:
                env.render()

        returns.append(total_reward)
        print(f"Episode {ep + 1}: Total Reward = {total_reward:.2f} in {step} steps")

    avg_return = np.mean(returns)
    print(f"\nAverage return over {episodes} episodes: {avg_return:.2f}")


if __name__ == "__main__":
    args = parse_args()
    print(args)

    # Generate run_name from arguments
    run_name = f"{args.gym_id}__{args.exp_name}__{args.seed}__{int(time.time())}"

    if args.track:
        import wandb

        wandb.init(
            project=args.derek_project_name,
            entity=args.derek_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=False,
            save_code=True,
        )

    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text("hyperparameters",
                    "param_value\n%s" % ("\n".join(f"{key}!{value}" for key, value in vars(args).items())))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Pass run_name when creating environments
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.gym_id, args.seed + i, i, args.capture_video, run_name, price_data) for i in
         range(args.num_envs)]
    )

    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"
    # print("envs.single_observation_space", envs.single_observation_space.shape)
    # print("envs.action_space", envs.single_action_space.n)
    max_cardinality = 10
    min_weight = 0.0  # Lower bound for asset allocation
    max_weight = 1.0
    agent = Agent(envs, max_cardinality=max_cardinality,min_cardinality=max_cardinality, min_weight=min_weight, max_weight=max_weight).to(device)
    # print(agent)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate,eps=1e-5)
    obs=torch.zeros((args.num_steps,args.num_envs)+envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps,args.num_envs)+envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps,args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps,args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps,args.num_envs)).to(device)
    values = torch.zeros((args.num_steps,args.num_envs)).to(device)

    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset()
    next_obs = torch.tensor(next_obs, dtype=torch.float32).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    num_updates = args.total_timesteps//args.batch_size
    # print(num_updates)
    # print("next_obs.shape",next_obs.shape)
    # print("agent.get_value(next_obs)",agent.get_value(next_obs))
    # print("agent.get_value(next_obs).shape)",agent.get_value(next_obs).shape)
    # print()
    # print("agent.get_action_and_value(next_obs)",agent.get_action_and_value(next_obs))

    portfolio_return = 0  # This will track the total portfolio return


    for update in range(1, num_updates + 1):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += 1 * args.num_envs

            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()

            actions[step] = action
            logprobs[step] = logprob

            # Correct action step
            next_obs_np, reward, terminated, truncated, info = envs.step(action.cpu().numpy().reshape(1, -1))
            done = np.logical_or(terminated, truncated)

            rewards[step] = torch.tensor(reward).to(device).view(-1)

            # Track portfolio return by accumulating the reward
            portfolio_return += reward  # Sum up the portfolio returns

            next_obs = torch.tensor(next_obs_np, dtype=torch.float32).to(device)
            next_done = torch.tensor(done, dtype=torch.float32).to(device)

            # Track episodic return
            for item in info:
                if "episode" in item:
                    episode_indices = np.where(info['episode']['_r'] == True)[0][0]
                    print(f"global_step={global_step}, episodic_return={info['episode']['r'][episode_indices]}")
                    writer.add_scalar("charts/episodic_return", info["episode"]["r"][episode_indices], global_step)
                    writer.add_scalar("charts/episodic_length", info["episode"]["l"][episode_indices], global_step)

                    # Log portfolio return as a separate scalar
                    writer.add_scalar("charts/portfolio_return", portfolio_return, global_step)

                    if args.track and args.capture_video:
                        import glob

                        video_files = sorted(glob.glob(f"videos/{run_name}/*.mp4"))
                        if video_files:
                            latest_video = video_files[-1]
                            wandb.log({"video": wandb.Video(latest_video, caption=f"Step {global_step}", fps=4,
                                                            format="mp4")}, step=global_step)

                    break

        # bootstrap value
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            if args.gae:
                advantages = torch.zeros_like(rewards).to(device)
                lastgaelam = 0
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        nextvalues = values[t + 1]
                    delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                    advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                returns = advantages + values
            else:
                returns = torch.zeros_like(rewards).to(device)
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        next_return = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        next_return = returns[t + 1]
                    returns[t] = rewards[t] + args.gamma * nextnonterminal * next_return
                advantages = returns - values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():

                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), max_norm=1.0)
                optimizer.step()

            if args.target_kl is not None:
                if approx_kl > args.target_kl:
                    break

            y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

            # TRY NOT TO MODIFY: record rewards for plotting purposes
            writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
            writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
            writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
            writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
            writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
            writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
            writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
            writer.add_scalar("losses/explained_variance", explained_var, global_step)
            # print("SPS:", int(global_step / (time.time() - start_time)))
            writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    # Save model
    # torch.save(agent.state_dict(), "ppo_portfolio_agent_stocks.pt")

    # Recreate environment for evaluation (no wrappers if you want full control)
    eval_env = PortfolioEnv(price_data=price_data)
    agent.load_state_dict(torch.load("ppo_portfolio_agent_stocks.pt", map_location=device))

    # Evaluate the trained agent
    evaluate(agent, eval_env, device, episodes=20, render=False)


    # === Evaluate a fixed equally weighted portfolio (2% per asset) ===
    fixed_weights = np.array([1/50] * eval_env.n_assets)
    print(eval_env.n_assets)
    fixed_portfolio_value = eval_env.initial_value
    fixed_returns = []

    # === Track PPO Agent portfolio value based on selected weights ===
    ppo_weights = []
    ppo_portfolio_values = []
    ppo_value = eval_env.initial_value
    obs, _ = eval_env.reset()

    for t in range(eval_env.max_steps - 1):
        obs_tensor = torch.tensor(obs, dtype=torch.float32).to(device)
        with torch.no_grad():
            action, _, _, _ = agent.get_action_and_value(obs_tensor)

        # Save weights
        weights = action.cpu().numpy().flatten()
        weights = weights / (np.sum(weights) + 1e-8)
        ppo_weights.append(weights)

        # Get price relatives
        prev_prices = eval_env.price_data.iloc[t]
        curr_prices = eval_env.price_data.iloc[t + 1]
        price_relatives = curr_prices / prev_prices

        # Calculate weighted return
        step_return = np.dot(weights, price_relatives)
        ppo_value *= step_return

        # Clip the portfolio value to avoid extreme values
        ppo_value = np.clip(ppo_value, 1e-8, 1e8)

        ppo_portfolio_values.append(ppo_value)

        # Step env to update observation
        obs, _, done, _, _ = eval_env.step(weights)

    # Create fixed portfolio value over time
    fixed_portfolio_values = []
    fixed_value = eval_env.initial_value
    for t in range(eval_env.max_steps - 1):
        prev_prices = eval_env.price_data.iloc[t]
        curr_prices = eval_env.price_data.iloc[t + 1]
        price_relatives = curr_prices / prev_prices
        step_return = np.dot(fixed_weights, price_relatives)
        fixed_value *= step_return
        fixed_portfolio_values.append(fixed_value)

    # Plot evolution of portfolio values
    plt.figure(figsize=(12, 6))
    plt.plot(ppo_portfolio_values, label="Trained PPO Portfolio Constrained train", linewidth=2)
    plt.plot(fixed_portfolio_values, label="Fixed 2% Portfolio", linestyle="--", linewidth=2)

    plt.title("Portfolio Value Over Time")
    plt.xlabel("Time Step")
    plt.ylabel("Portfolio Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Convert portfolio values to numpy arrays
    ppo_values = np.array(ppo_portfolio_values)
    fixed_values = np.array(fixed_portfolio_values)

    # Calculate daily returns
    ppo_returns = ppo_values[1:] / ppo_values[:-1] - 1
    fixed_returns = fixed_values[1:] / fixed_values[:-1] - 1


    # Calculate metrics
    def compute_metrics(returns, values):
        total_return = values[-1] / values[0] - 1
        volatility = np.std(returns)
        sharpe_ratio = np.mean(returns) / (volatility + 1e-8)
        return total_return, volatility, sharpe_ratio


    ppo_ret, ppo_vol, ppo_sharpe = compute_metrics(ppo_returns, ppo_values)
    fixed_ret, fixed_vol, fixed_sharpe = compute_metrics(fixed_returns, fixed_values)

    # Create results table
    results = pd.DataFrame({
        "Portfolio": ["PPO Agent contstrained train", "Fixed 2%"],
        "Return": [ppo_ret, fixed_ret],
        "Volatility": [ppo_vol, fixed_vol],
        "Sharpe Ratio": [ppo_sharpe, fixed_sharpe]
    })

    # Format as percentage
    results["Return"] = results["Return"] * 100
    results["Volatility"] = results["Volatility"] * 100
    results["Sharpe Ratio"] = results["Sharpe Ratio"]

    # Display the table
    print(results.to_string(index=False, float_format="%.2f"))

    # Download new data
    data = yf.download(stocks, start="2024-01-02", end="2025-04-02")
    data = data["Close"].reset_index()
    data.to_csv("test_stock_prices_2024.csv", index=False)
    price_data = pd.read_csv("test_stock_prices_2024.csv")
    price_data = price_data.drop(columns="Date")
    # Recreate environment for evaluation (no wrappers if you want full control)
    eval_env = PortfolioEnv(price_data=price_data)
    agent.load_state_dict(torch.load("ppo_portfolio_agent_stocks.pt", map_location=device))

    # Evaluate the trained agent
    evaluate(agent, eval_env, device, episodes=20, render=False)

    # === Evaluate a fixed equally weighted portfolio (25% per asset) ===
    fixed_weights = np.array([1 / 50] * eval_env.n_assets)
    print(eval_env.n_assets)
    fixed_portfolio_value = eval_env.initial_value
    fixed_returns = []

    # === Track PPO Agent portfolio value based on selected weights ===
    ppo_weights = []
    ppo_portfolio_values = []
    ppo_value = eval_env.initial_value
    obs, _ = eval_env.reset()

    for t in range(eval_env.max_steps - 1):
        obs_tensor = torch.tensor(obs, dtype=torch.float32).to(device)
        with torch.no_grad():
            action, _, _, _ = agent.get_action_and_value(obs_tensor)

        # Save weights
        weights = action.cpu().numpy().flatten()
        weights = weights / (np.sum(weights) + 1e-8)
        ppo_weights.append(weights)

        # Get price relatives
        prev_prices = eval_env.price_data.iloc[t]
        curr_prices = eval_env.price_data.iloc[t + 1]
        price_relatives = curr_prices / prev_prices

        # Calculate weighted return
        step_return = np.dot(weights, price_relatives)
        ppo_value *= step_return

        # Clip the portfolio value to avoid extreme values
        ppo_value = np.clip(ppo_value, 1e-8, 1e8)

        ppo_portfolio_values.append(ppo_value)

        # Step env to update observation
        obs, _, done, _, _ = eval_env.step(weights)

    # Create fixed portfolio value over time
    fixed_portfolio_values = []
    fixed_value = eval_env.initial_value
    for t in range(eval_env.max_steps - 1):
        prev_prices = eval_env.price_data.iloc[t]
        curr_prices = eval_env.price_data.iloc[t + 1]
        price_relatives = curr_prices / prev_prices
        step_return = np.dot(fixed_weights, price_relatives)
        fixed_value *= step_return
        fixed_portfolio_values.append(fixed_value)

    # Plot evolution of portfolio values
    plt.figure(figsize=(12, 6))
    plt.plot(ppo_portfolio_values, label="Trained PPO Portfolio Constrained test", linewidth=2)
    plt.plot(fixed_portfolio_values, label="Fixed 2% Portfolio", linestyle="--", linewidth=2)

    plt.title("Portfolio Value Over Time")
    plt.xlabel("Time Step")
    plt.ylabel("Portfolio Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Convert portfolio values to numpy arrays
    ppo_values = np.array(ppo_portfolio_values)
    fixed_values = np.array(fixed_portfolio_values)

    # Calculate daily returns
    ppo_returns = ppo_values[1:] / ppo_values[:-1] - 1
    fixed_returns = fixed_values[1:] / fixed_values[:-1] - 1


    # Calculate metrics
    def compute_metrics(returns, values):
        total_return = values[-1] / values[0] - 1
        volatility = np.std(returns)
        sharpe_ratio = np.mean(returns) / (volatility + 1e-8)
        return total_return, volatility, sharpe_ratio


    ppo_ret, ppo_vol, ppo_sharpe = compute_metrics(ppo_returns, ppo_values)
    fixed_ret, fixed_vol, fixed_sharpe = compute_metrics(fixed_returns, fixed_values)

    # Create results table
    results = pd.DataFrame({
        "Portfolio": ["PPO Agent contrainted test", "Fixed 2%"],
        "Return": [ppo_ret, fixed_ret],
        "Volatility": [ppo_vol, fixed_vol],
        "Sharpe Ratio": [ppo_sharpe, fixed_sharpe]
    })

    # Format as percentage
    results["Return"] = results["Return"] * 100
    results["Volatility"] = results["Volatility"] * 100
    results["Sharpe Ratio"] = results["Sharpe Ratio"]

    # Display the table
    print(results.to_string(index=False, float_format="%.2f"))


    envs.close()
    writer.close()

