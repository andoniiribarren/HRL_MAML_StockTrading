from typing import List
import gymnasium as gym
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gymnasium import spaces
from gymnasium.utils import seeding
from stable_baselines3.common.vec_env import DummyVecEnv

matplotlib.use("Agg")


class StockTradingEnvHRL(gym.Env):
    """
    A stock trading environment for OpenAI gym

    Parameters:
        df (pandas.DataFrame): Dataframe containing data
        hmax (int): Maximum cash to be traded in each trade per asset.
        initial_amount (int): Amount of cash initially available
        buy_cost_pct (float, array): Cost for buying shares, each index corresponds to each asset
        sell_cost_pct (float, array): Cost for selling shares, each index corresponds to each asset
        turbulence_threshold (float): Maximum turbulence allowed in market for purchases to occur. If exceeded, positions are liquidated
        print_verbosity(int): When iterating (step), how often to print stats about state of env
    """

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        df: pd.DataFrame,
        stock_dim: int,
        hmax: int,
        initial_amount: int,
        num_stock_shares: list[int],
        buy_cost_pct: list[float],
        sell_cost_pct: list[float],
        state_space_M: int,
        state_space_W: int,
        action_space: int,
        tech_indicator_list: list[str],
        make_plots: bool = False,
        print_verbosity=10,
        day=0,
        initial=True,
        previous_state=[],
        model_name="",
        mode="",
        iteration="",
    ):
        self.day = day
        self.df = df
        self.stock_dim = stock_dim
        self.hmax = hmax
        self.num_stock_shares = num_stock_shares
        self.initial_amount = initial_amount  # get the initial cash
        self.buy_cost_pct = buy_cost_pct
        self.sell_cost_pct = sell_cost_pct
        self.state_space_M = state_space_M
        self.state_space_W = state_space_W
        self.action_space = action_space
        self.tech_indicator_list = tech_indicator_list

        self.action_space = spaces.Box(low=-1, high=1, shape=(self.action_space,))

        full_state_space = (
            1 + 2 * self.stock_dim + len(self.tech_indicator_list) * self.stock_dim
        )

        self.observation_space = spaces.Dict(
            {
                "full_state": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(
                        full_state_space,
                    ),  # 1 + 2 * stock_dim + len(INDICATORS) * stock_dim
                    dtype=np.float32,
                ),
                "manager": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(
                        self.state_space_M,
                    ),  # stock_dim + len(INDICATORS) * stock_dim
                    dtype=np.float32,
                ),
                "worker": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(self.state_space_W,),  # 1+ 3 * stock_dim
                    dtype=np.float32,
                ),
            }
        )

        self.data = self.df[self.df.dayorder == self.day]
        self.terminal = False
        self.make_plots = make_plots
        self.print_verbosity = print_verbosity
        self.initial = initial
        self.previous_state = previous_state
        self.model_name = model_name
        self.mode = mode
        self.iteration = iteration
        self.SHARE_SCALE = 2000
        # initalize state
        self.state = self._initiate_state()

        # initialize reward
        self.reward = 0
        self.cost = 0
        self.trades = 0
        self.episode = 0
        # memorize all the total balance change
        self.asset_memory = [
            self.initial_amount
            + np.sum(
                np.array(self.num_stock_shares)
                * np.array(self.state["full_state"][1 : 1 + self.stock_dim])
            )
        ]  # the initial total asset is calculated by cash + sum (num_share_stock_i * price_stock_i)
        self.rewards_memory = []
        self.actions_memory = []
        self.state_memory = (
            []
        )  # we need sometimes to preserve the state in the middle of trading process
        self.date_memory = [self._get_date()]
        #         self.logger = Logger('results',[CSVOutputFormat])
        # self.reset()

        self._seed()

    def _sell_stock(self, index, action):
        def _do_sell_normal():

            if self.state["full_state"][index + self.stock_dim + 1] > 0:
                # Sell only if current asset is > 0
                sell_num_shares = min(
                    abs(action),
                    self.state["full_state"][index + self.stock_dim + 1],
                )
                sell_amount = (
                    self.state["full_state"][index + 1]
                    * sell_num_shares
                    * (1 - self.sell_cost_pct[index])
                )
                # update balance
                self.state["full_state"][0] += sell_amount
                self.state["worker"][0] += sell_amount / self.initial_amount

                # Update number of stocks
                self.state["full_state"][index + self.stock_dim + 1] -= sell_num_shares
                self.state["worker"][index + self.stock_dim + 1] -= (
                    sell_num_shares / self.SHARE_SCALE
                )
                self.cost += (
                    self.state["full_state"][index + 1]
                    * sell_num_shares
                    * self.sell_cost_pct[index]
                )
                self.trades += 1
            else:
                sell_num_shares = 0

            return sell_num_shares

        # perform sell action based on the sign of the action
        sell_num_shares = _do_sell_normal()

        return sell_num_shares

    def _buy_stock(self, index, action):
        def _do_buy():
            available_amount = self.state["full_state"][0] // (
                self.state["full_state"][index + 1] * (1 + self.buy_cost_pct[index])
            )  # when buying stocks, we should consider the cost of trading when calculating available_amount, or we may be have cash<0
            # print('available_amount:{}'.format(available_amount))

            # update balance
            buy_num_shares = min(available_amount, action)
            buy_amount = (
                self.state["full_state"][index + 1]
                * buy_num_shares
                * (1 + self.buy_cost_pct[index])
            )
            self.state["full_state"][0] -= buy_amount
            self.state["worker"][0] -= buy_amount / self.initial_amount

            self.state["full_state"][index + self.stock_dim + 1] += buy_num_shares
            self.state["worker"][index + self.stock_dim + 1] += (
                buy_num_shares / self.SHARE_SCALE
            )

            self.cost += (
                self.state["full_state"][index + 1]
                * buy_num_shares
                * self.buy_cost_pct[index]
            )
            self.trades += 1

            return buy_num_shares

        # perform buy action based on the sign of the action
        buy_num_shares = _do_buy()

        return buy_num_shares

    def _make_plot(self):

        dates = pd.to_datetime(self.date_memory)
        agent_curve = np.array(self.asset_memory)
        df = self.df
        df_ep = df[df["dayorder"].isin(range(len(dates)))]
        price_matrix = df_ep.pivot_table(
            index="dayorder", columns="tic", values="close"
        ).sort_index()
        initial_prices = price_matrix.iloc[0].values
        bh_curve = (price_matrix.values / initial_prices).mean(
            axis=1
        ) * self.initial_amount

        plt.figure(figsize=(12, 6))
        plt.plot(dates, agent_curve, label="Agente RL", color="red")
        plt.plot(dates, bh_curve, label="Buy & Hold", color="blue", linestyle="--")
        plt.xlabel("Fecha")
        plt.ylabel("Valor de la cartera")
        plt.title("Evolución de portfolio: RL vs Buy & Hold")
        plt.legend()
        plt.savefig(f"results/account_value_trade_{self.episode}.png")
        plt.close()

    def step(self, actions):
        self.terminal = self.day >= self.df["dayorder"].nunique() - 1
        if self.terminal:
            # print(f"Episode: {self.episode}")
            # print("DEBUG 1: ", self.state)
            if self.make_plots:
                self._make_plot()
            end_total_asset = self.state["full_state"][0] + sum(
                np.array(self.state["full_state"][1 : (self.stock_dim + 1)])
                * np.array(
                    self.state["full_state"][
                        (self.stock_dim + 1) : (self.stock_dim * 2 + 1)
                    ]
                )
            )
            df_total_value = pd.DataFrame(self.asset_memory)
            tot_reward = (
                self.state["full_state"][0]
                + sum(
                    np.array(self.state["full_state"][1 : (self.stock_dim + 1)])
                    * np.array(
                        self.state["full_state"][
                            (self.stock_dim + 1) : (self.stock_dim * 2 + 1)
                        ]
                    )
                )
                - self.asset_memory[0]
            )  # initial_amount is only cash part of our initial asset
            df_total_value.columns = ["account_value"]
            df_total_value["date"] = self.date_memory
            df_total_value["daily_return"] = df_total_value["account_value"].pct_change(
                1
            )
            if df_total_value["daily_return"].std() != 0:
                sharpe = (
                    (252**0.5)
                    * df_total_value["daily_return"].mean()
                    / df_total_value["daily_return"].std()
                )
            df_rewards = pd.DataFrame(self.rewards_memory)
            df_rewards.columns = ["account_rewards"]
            df_rewards["date"] = self.date_memory[:-1]
            if self.episode % self.print_verbosity == 0:
                print(f"day: {self.day}, episode: {self.episode}")
                print(f"begin_total_asset: {self.asset_memory[0]:0.2f}")
                print(f"end_total_asset: {end_total_asset:0.2f}")
                print(f"total_reward: {tot_reward:0.2f}")
                print(f"total_cost: {self.cost:0.2f}")
                print(f"total_trades: {self.trades}")
                if df_total_value["daily_return"].std() != 0:
                    print(f"Sharpe: {sharpe:0.3f}")
                print("=================================")

            if (self.model_name != "") and (self.mode != ""):
                df_actions = self.save_action_memory()
                df_actions.to_csv(
                    "results/actions_{}_{}_{}.csv".format(
                        self.mode, self.model_name, self.iteration
                    )
                )
                df_total_value.to_csv(
                    "results/account_value_{}_{}_{}.csv".format(
                        self.mode, self.model_name, self.iteration
                    ),
                    index=False,
                )
                df_rewards.to_csv(
                    "results/account_rewards_{}_{}_{}.csv".format(
                        self.mode, self.model_name, self.iteration
                    ),
                    index=False,
                )
                plt.plot(self.asset_memory, "r")
                plt.savefig(
                    "results/account_value_{}_{}_{}.png".format(
                        self.mode, self.model_name, self.iteration
                    )
                )
                plt.close()

            # Add outputs to logger interface
            # logger.record("environment/portfolio_value", end_total_asset)
            # logger.record("environment/total_reward", tot_reward)
            # logger.record("environment/total_reward_pct", (tot_reward / (end_total_asset - tot_reward)) * 100)
            # logger.record("environment/total_cost", self.cost)
            # logger.record("environment/total_trades", self.trades)

            return (
                self.state,
                self.reward,
                self.terminal,
                False,
                {},
            )

        else:
            actions = actions * self.hmax  # actions initially is scaled between 0 to 1
            actions = actions.astype(int)
            begin_total_asset = self.state["full_state"][0] + sum(
                np.array(self.state["full_state"][1 : (self.stock_dim + 1)])
                * np.array(
                    self.state["full_state"][
                        (self.stock_dim + 1) : (self.stock_dim * 2 + 1)
                    ]
                )
            )
            # print("begin_total_asset:{}".format(begin_total_asset))
            begin_prices = np.array(self.state["full_state"][1 : (self.stock_dim + 1)])

            argsort_actions = np.argsort(actions)
            sell_index = argsort_actions[: np.where(actions < 0)[0].shape[0]]
            buy_index = argsort_actions[::-1][: np.where(actions > 0)[0].shape[0]]

            for index in sell_index:
                # print(f"Num shares before: {self.state[index+self.stock_dim+1]}")
                # print(f'take sell action before : {actions[index]}')
                actions[index] = self._sell_stock(index, actions[index]) * (-1)
                # print(f'take sell action after : {actions[index]}')
                # print(f"Num shares after: {self.state[index+self.stock_dim+1]}")

            for index in buy_index:
                # print('take buy action: {}'.format(actions[index]))
                actions[index] = self._buy_stock(index, actions[index])

            self.actions_memory.append(actions)
            # print(actions)

            # state: s -> s+1
            self.day += 1
            self.data = self.df[self.df.dayorder == self.day]
            self.state = self._update_state()

            end_total_asset = self.state["full_state"][0] + sum(
                np.array(self.state["full_state"][1 : (self.stock_dim + 1)])
                * np.array(
                    self.state["full_state"][
                        (self.stock_dim + 1) : (self.stock_dim * 2 + 1)
                    ]
                )
            )

            end_prices = np.array(self.state["full_state"][1 : (self.stock_dim + 1)])

            self.asset_memory.append(end_total_asset)
            self.date_memory.append(self._get_date())

            self.reward = np.log(end_total_asset / begin_total_asset)
            self.rewards_memory.append(self.reward)

            self.state_memory.append(
                self.state
            )  # add current state in state_recorder for each step

            # ALIGNMENT REWARD
            price_change = end_prices - begin_prices

            rew_align = np.where(
                actions == 0, 0.0, np.sign(actions) * np.sign(price_change)
            ).astype(np.float32)

            info = {"rew_align": rew_align}

        return self.state, self.reward, self.terminal, False, info

    def reset(
        self,
        *,
        seed=None,
        options=None,
    ):
        # initiate state
        self.day = 0
        self.data = self.df[self.df.dayorder == self.day]
        self.state = self._initiate_state()

        if self.initial:
            self.asset_memory = [
                self.initial_amount
                + np.sum(
                    np.array(self.num_stock_shares)
                    * np.array(self.state["full_state"][1 : 1 + self.stock_dim])
                )
            ]
        else:
            previous_total_asset = self.previous_state["full_state"][0] + sum(
                np.array(self.state["full_state"][1 : (self.stock_dim + 1)])
                * np.array(
                    self.previous_state["full_state"][
                        (self.stock_dim + 1) : (self.stock_dim * 2 + 1)
                    ]
                )
            )
            self.asset_memory = [previous_total_asset]

        self.cost = 0
        self.trades = 0
        self.terminal = False
        # self.iteration=self.iteration
        self.rewards_memory = []
        self.actions_memory = []
        self.date_memory = [self._get_date()]

        self.episode += 1

        return self.state, {}

    def render(self, mode="human", close=False):
        # return self.state
        raise NotImplementedError()

    def _initiate_state(self):
        if self.initial:
            # For Initial State
            if len(self.df.tic.unique()) > 1:
                # for multiple stock
                state = {
                    "full_state": (
                        [self.initial_amount]  # balance inicial
                        + self.data.close_raw.values.tolist()  # Close prices i
                        + self.num_stock_shares  # stock shares i
                        + sum(
                            (
                                self.data[tech].values.tolist()
                                for tech in self.tech_indicator_list  # Indicators i
                            ),
                            [],
                        )
                    ),
                    "manager": self.data.close.values.tolist()
                    + sum(
                        (
                            self.data[tech].values.tolist()
                            for tech in self.tech_indicator_list
                        ),
                        [],
                    ),
                    "worker": [self.initial_amount / self.initial_amount]
                    + self.data.close.values.tolist()
                    + [s / self.SHARE_SCALE for s in self.num_stock_shares]
                    + [0] * self.stock_dim,
                }

            else:
                # for single stock
                raise ValueError(
                    "No implemented for single stock. Please, select more than one ticker."
                )
        else:
            # Using Previous State
            if len(self.df.tic.unique()) > 1:
                # for multiple stock
                state = {
                    "full_state": (
                        [self.previous_state["full_state"][0]]
                        + self.data.close_raw.values.tolist()
                        + self.previous_state["full_state"][
                            (self.stock_dim + 1) : (self.stock_dim * 2 + 1)
                        ]
                        + sum(
                            (
                                self.data[tech].values.tolist()
                                for tech in self.tech_indicator_list
                            ),
                            [],
                        )
                    ),
                    "manager": (
                        self.data.close.values.tolist()
                        + sum(
                            (
                                self.data[tech].values.tolist()
                                for tech in self.tech_indicator_list
                            ),
                            [],
                        )
                    ),
                    "worker": (
                        [self.previous_state["worker"][0]]
                        + self.data.close.values.tolist()
                        + self.previous_state["worker"][
                            (self.stock_dim + 1) : (self.stock_dim * 2 + 1)
                        ]
                        + self.previous_state["worker"][
                            (self.stock_dim * 2 + 1) : (self.stock_dim * 3 + 1)
                        ]
                    ),
                }
            else:
                # for single stock
                raise ValueError(
                    "No implemented for single stock. Please, select more than one ticker."
                )
        return state

    def _update_state(self):
        if len(self.df.tic.unique()) > 1:
            # for multiple stock
            state = {
                "full_state": (
                    [self.state["full_state"][0]]
                    + self.data.close_raw.values.tolist()
                    + list(
                        self.state["full_state"][
                            (self.stock_dim + 1) : (self.stock_dim * 2 + 1)
                        ]
                    )
                    + sum(
                        (
                            self.data[tech].values.tolist()
                            for tech in self.tech_indicator_list
                        ),
                        [],
                    )
                ),
                "manager": (
                    self.data.close.values.tolist()
                    + sum(
                        (
                            self.data[tech].values.tolist()
                            for tech in self.tech_indicator_list
                        ),
                        [],
                    )
                ),
                "worker": (
                    [self.state["worker"][0]]
                    + self.data.close.values.tolist()
                    + list(
                        self.state["worker"][
                            (self.stock_dim + 1) : (self.stock_dim * 2 + 1)
                        ]
                    )
                    + [0]
                    * self.stock_dim  # Añadimos vector de acciones del manager que luego será reemplazado en train
                ),
            }

        else:
            # for single stock
            raise ValueError(
                "No implemented for single stock. Please, select more than one ticker."
            )

        return state

    def _get_date(self):
        if len(self.df.tic.unique()) > 1:
            date = self.data.date.unique()[0]
        else:
            date = self.data.date
        return date

    # add save_state_memory to preserve state in the trading process
    def save_state_memory(self):
        if len(self.df.tic.unique()) > 1:
            # date and close price length must match actions length
            date_list = self.date_memory[:-1]
            df_date = pd.DataFrame(date_list)
            df_date.columns = ["date"]

            state_list = self.state_memory
            df_states = pd.DataFrame(
                state_list,
                columns=[
                    "cash",
                    "Bitcoin_price",
                    "Gold_price",
                    "Bitcoin_num",
                    "Gold_num",
                    "Bitcoin_Disable",
                    "Gold_Disable",
                ],
            )
            df_states.index = df_date.date
            # df_actions = pd.DataFrame({'date':date_list,'actions':action_list})
        else:
            raise ValueError(
                "No implemented for single stock. Please, select more than one ticker."
            )
        return df_states

    def save_asset_memory(self):
        date_list = self.date_memory
        asset_list = self.asset_memory
        # print(len(date_list))
        # print(len(asset_list))
        df_account_value = pd.DataFrame(
            {"date": date_list, "account_value": asset_list}
        )
        return df_account_value

    def save_action_memory(self):
        if len(self.df.tic.unique()) > 1:
            # date and close price length must match actions length
            date_list = self.date_memory[:-1]
            df_date = pd.DataFrame(date_list)
            df_date.columns = ["date"]

            action_list = self.actions_memory
            df_actions = pd.DataFrame(action_list)
            df_actions.columns = self.data.tic.values
            df_actions.index = df_date.date
            # df_actions = pd.DataFrame({'date':date_list,'actions':action_list})
        else:
            date_list = self.date_memory[:-1]
            action_list = self.actions_memory
            df_actions = pd.DataFrame({"date": date_list, "actions": action_list})
        return df_actions

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def get_sb_env(self):
        e = DummyVecEnv([lambda: self])
        obs = e.reset()
        return e, obs
