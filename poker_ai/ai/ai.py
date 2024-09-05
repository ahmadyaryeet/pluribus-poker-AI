import copy
import logging
import multiprocessing as mp
import os
from pathlib import Path
from typing import Dict, List, Union
from datetime import datetime
import sys

import joblib
import numpy as np

from poker_ai.ai.agent import Agent
from poker_ai.games.short_deck.state import ShortDeckPokerState


log = logging.getLogger("sync.ai")

def log_to_console(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"{timestamp} - {message}", file=sys.stderr, flush=True)

def calculate_strategy(this_info_sets_regret: Dict[str, float]) -> Dict[str, float]:
    """
    Calculate the strategy based on the current information sets regret.

    Parameters
    ----------
    this_info_sets_regret : Dict[str, float]
        Regret for each action at this info set.

    Returns
    -------
    strategy : Dict[str, float]
        Strategy as a probability over actions.
    """
    #log_to_console(f"Calculating strategy for info set regret: {this_info_sets_regret}")
    
    actions = this_info_sets_regret.keys()
    regret_sum = sum([max(regret, 0) for regret in this_info_sets_regret.values()])
    
    #log_to_console(f"Regret sum: {regret_sum}")
    
    if regret_sum > 0:
        strategy = {
            action: max(this_info_sets_regret[action], 0) / regret_sum
            for action in actions
        }
        #log_to_console(f"Calculated strategy (regret sum > 0): {strategy}")
    else:
        strategy = {action: 1.0 / len(actions) for action in actions}
        #log_to_console(f"Using default strategy (regret sum <= 0): {strategy}")
    
    return strategy


def update_strategy(
    agent: Agent,
    state: ShortDeckPokerState,
    i: int,
    t: int,
    locks: Dict[str, mp.synchronize.Lock] = {},
):
    """
    Update pre flop strategy using a more theoretically sound approach.

    ...

    Parameters
    ----------
    agent : Agent
        Agent being trained.
    state : ShortDeckPokerState
        Current game state.
    i : int
        The Player.
    t : int
        The iteration.
    locks : Dict[str, mp.synchronize.Lock]
        The locks for multiprocessing
    """
    ph = state.player_i  # this is always the case no matter what i is

    player_not_in_hand = not state.players[i].is_active
    if state.is_terminal or player_not_in_hand or state.betting_round > 0:
        return

    # NOTE(fedden): According to Algorithm 1 in the supplementary material,
    #               we would add in the following bit of logic. However we
    #               already have the game logic embedded in the state class,
    #               and this accounts for the chance samplings. In other words,
    #               it makes sure that chance actions such as dealing cards
    #               happen at the appropriate times.
    # elif h is chance_node:
    #   sample action from strategy for h
    #   update_strategy(rs, h + a, i, t)

    elif ph == i:
        # calculate regret
        this_info_sets_regret = agent.regret.get(state.info_set, state.initial_regret)
        sigma = calculate_strategy(this_info_sets_regret)
        log.debug(f"Calculated Strategy for {state.info_set}: {sigma}")
        # choose an action based of sigma
        available_actions: List[str] = list(sigma.keys())
        action_probabilities: np.ndarray = np.array(list(sigma.values()))
        action: str = np.random.choice(available_actions, p=action_probabilities)
        log.debug(f"ACTION SAMPLED: ph {state.player_i} ACTION: {action}")
        # Increment the action counter.
        if locks:
            locks["strategy"].acquire()
        this_states_strategy = agent.strategy.get(
            state.info_set, state.initial_strategy
        )
        this_states_strategy[action] += 1
        # Update the master strategy by assigning.
        agent.strategy[state.info_set] = this_states_strategy
        if locks:
            locks["strategy"].release()
        new_state: ShortDeckPokerState = state.apply_action(action)
        update_strategy(agent, new_state, i, t, locks)
    else:
        # Traverse each action.
        for action in state.legal_traverse_actions:
            log.debug(f"Going to Traverse {action} for opponent")
            new_state: ShortDeckPokerState = state.apply_action(action)
            update_strategy(agent, new_state, i, t, locks)

def log_training_metrics(agent, iteration):
    if iteration % 50 == 0:  # Log every 1000 iterations
        avg_regret = sum(abs(r) for regrets in agent.regret.values() for r in regrets.values()) / len(agent.regret)
        log_to_console(f"Iteration {iteration}: Average absolute regret: {avg_regret}")

        total_entropy = 0
        for info_set, regrets in agent.regret.items():
            strategy = calculate_strategy(regrets)
            entropy = -sum(p * np.log2(p) for p in strategy.values() if p > 0)
            total_entropy += entropy
        avg_entropy = total_entropy / len(agent.regret)
        log_to_console(f"Iteration {iteration}: Average strategy entropy: {avg_entropy}")

def cfr(
    agent: Agent,
    state: ShortDeckPokerState,
    i: int,
    t: int,
    locks: Dict[str, mp.synchronize.Lock] = {},
) -> float:
    """
    Regular counterfactual regret minimization algorithm.
    """
    log_training_metrics(agent, t)  # This will now only log every 100 iterations

    current_player = state.player_i

    if state.is_terminal or not state.players[i].is_active:
        return state.payout[i]

    if current_player == i:
        this_info_sets_regret = agent.regret.get(state.info_set, state.initial_regret)
        sigma = calculate_strategy(this_info_sets_regret)

        vo = 0.0
        voa: Dict[str, float] = {}
        for action in state.legal_traverse_actions:
            new_state: ShortDeckPokerState = state.apply_action(action)
            voa[action] = cfr(agent, new_state, i, t, locks)
            vo += sigma[action] * voa[action]

        if locks:
            locks["regret"].acquire()
        
        this_info_sets_regret = agent.regret.get(state.info_set, state.initial_regret)
        for action in state.legal_traverse_actions:
            this_info_sets_regret[action] += voa[action] - vo
        
        agent.regret[state.info_set] = this_info_sets_regret
        
        if locks:
            locks["regret"].release()
        
        return vo
    else:
        this_info_sets_regret = agent.regret.get(state.info_set, state.initial_regret)
        sigma = calculate_strategy(this_info_sets_regret)
        available_actions: List[str] = list(sigma.keys())
        action_probabilities: List[float] = list(sigma.values())
        action: str = np.random.choice(available_actions, p=action_probabilities)
        new_state: ShortDeckPokerState = state.apply_action(action)
        return cfr(agent, new_state, i, t, locks)

def cfrp(
    agent: Agent,
    state: ShortDeckPokerState,
    i: int,
    t: int,
    c: int,
    locks: Dict[str, mp.synchronize.Lock] = {},
):
    """
    Counter factual regret minimazation with pruning.

    ...

    Parameters
    ----------
    agent : Agent
        Agent being trained.
    state : ShortDeckPokerState
        Current game state.
    i : int
        The Player.
    t : int
        The iteration.
    locks : Dict[str, mp.synchronize.Lock]
        The locks for multiprocessing
    """
    ph = state.player_i

    player_not_in_hand = not state.players[i].is_active
    if state.is_terminal or player_not_in_hand:
        return state.payout[i]
    # NOTE(fedden): The logic in Algorithm 1 in the supplementary material
    #               instructs the following lines of logic, but state class
    #               will already skip to the next in-hand player.
    # elif p_i not in hand:
    #   cfr()
    # NOTE(fedden): According to Algorithm 1 in the supplementary material,
    #               we would add in the following bit of logic. However we
    #               already have the game logic embedded in the state class,
    #               and this accounts for the chance samplings. In other words,
    #               it makes sure that chance actions such as dealing cards
    #               happen at the appropriate times.
    # elif h is chance_node:
    #   sample action from strategy for h
    #   cfr()
    elif ph == i:
        # calculate strategy
        this_info_sets_regret = agent.regret.get(state.info_set, state.initial_regret)
        sigma = calculate_strategy(this_info_sets_regret)
        # TODO: Does updating sigma here (as opposed to after regret) miss out
        #       on any updates? If so, is there any benefit to having it up
        #       here?
        vo = 0.0
        voa: Dict[str, float] = dict()
        # Explored dictionary to keep track of regret updates that can be
        # skipped.
        explored: Dict[str, bool] = {action: False for action in state.legal_traverse_actions}
        # Get the regret for this state.
        this_info_sets_regret = agent.regret.get(state.info_set, state.initial_regret)
        for action in state.legal_traverse_actions:
            if this_info_sets_regret[action] > c:
                new_state: ShortDeckPokerState = state.apply_action(action)
                voa[action] = cfrp(agent, new_state, i, t, c, locks)
                explored[action] = True
                vo += sigma[action] * voa[action]
        if locks:
            locks["regret"].acquire()
        # Get the regret for this state again, incase any other process updated
        # it whilst we were doing `cfrp`.
        this_info_sets_regret = agent.regret.get(state.info_set, state.initial_regret)
        for action in state.legal_traverse_actions:
            if explored[action]:
                this_info_sets_regret[action] += voa[action] - vo
        # Update the master copy of the regret.
        agent.regret[state.info_set] = this_info_sets_regret
        if locks:
            locks["regret"].release()
        return vo
    else:
        this_info_sets_regret = agent.regret.get(state.info_set, state.initial_regret)
        sigma = calculate_strategy(this_info_sets_regret)
        available_actions: List[str] = list(sigma.keys())
        action_probabilities: List[float] = list(sigma.values())
        action: str = np.random.choice(available_actions, p=action_probabilities)
        new_state: ShortDeckPokerState = state.apply_action(action)
        return cfrp(agent, new_state, i, t, c, locks)


def serialise(
    agent: Agent,
    save_path: Path,
    t: int,
    server_state: Dict[str, Union[str, float, int, None]],
    locks: Dict[str, mp.synchronize.Lock] = {},
):
    """
    Write progress of optimising agent (and server state) to file.

    ...

    Parameters
    ----------
    agent : Agent
        Agent being trained.
    save_path : ShortDeckPokerState
        Current game state.
    t : int
        The iteration.
    server_state : Dict[str, Union[str, float, int, None]]
        All the variables required to resume training.
    locks : Dict[str, mp.synchronize.Lock]
        The locks for multiprocessing
    """
    # Load the shared strategy that we accumulate into.
    agent_path = os.path.abspath(str(save_path / f"agent.joblib"))
    if os.path.isfile(agent_path):
        offline_agent = joblib.load(agent_path)
    else:
        offline_agent = {
            "regret": {},
            "timestep": t,
            "strategy": {},
            "pre_flop_strategy": {}
        }
    # Lock shared dicts so no other process modifies it whilst writing to
    # file.
    # Calculate the strategy for each info sets regret, and accumulate in
    # the offline agent's strategy.
    for info_set, this_info_sets_regret in sorted(agent.regret.items()):
        if locks:
            locks["regret"].acquire()
        strategy = calculate_strategy(this_info_sets_regret)
        if locks:
            locks["regret"].release()
        if info_set not in offline_agent["strategy"]:
            offline_agent["strategy"][info_set] = {
                action: probability for action, probability in strategy.items()
            }
        else:
            for action, probability in strategy.items():
                offline_agent["strategy"][info_set][action] += probability
    if locks:
        locks["regret"].acquire()
    offline_agent["regret"] = copy.deepcopy(agent.regret)
    if locks:
        locks["regret"].release()
    if locks:
        locks["pre_flop_strategy"].acquire()
    offline_agent["pre_flop_strategy"] = copy.deepcopy(agent.strategy)
    if locks:
        locks["pre_flop_strategy"].release()
    joblib.dump(offline_agent, agent_path)
    # Dump the server state to file too, but first update a few bits of the
    # state so when we load it next time, we start from the right place in
    # the optimisation process.
    server_path = save_path / f"server.gz"
    server_state["agent_path"] = agent_path
    server_state["start_timestep"] = t + 1
    joblib.dump(server_state, server_path)
