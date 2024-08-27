import random
import time
from typing import Dict

import click
import joblib
import numpy as np
from blessed import Terminal
import psutil
import os
import pickle

from poker_ai.games.short_deck.state import new_game, ShortDeckPokerState
from poker_ai.games.short_deck.self_play_state import new_self_play_game, SelfPlayShortDeckPokerState
from poker_ai.terminal.ascii_objects.card_collection import AsciiCardCollection
from poker_ai.terminal.ascii_objects.player import AsciiPlayer
from poker_ai.terminal.ascii_objects.logger import AsciiLogger
from poker_ai.terminal.render import print_footer, print_header, print_log, print_table
from poker_ai.terminal.results import UserResults
from poker_ai.utils.algos import rotate_list

def print_memory_usage():
    """Print the current memory usage."""
    process = psutil.Process()
    mem_info = process.memory_info()
    print(f"Memory Usage: {mem_info.rss / (1024 * 1024):.2f} MB")

def chunked_load(file_path, chunk_size=500*1024*1024):  # 500 MB chunks
    with open(file_path, 'rb') as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            yield chunk

def load_strategy(strategy_path):
    strategy = {}
    total_size = os.path.getsize(strategy_path)
    loaded_size = 0
    
    unpickler = pickle.Unpickler(open(strategy_path, 'rb'))
    
    for chunk in chunked_load(strategy_path):
        loaded_size += len(chunk)
        while chunk:
            try:
                part = unpickler.load()
                if isinstance(part, dict):
                    strategy.update(part)
                chunk = chunk[unpickler.tell():]
            except EOFError:
                break
        print(f"Loaded {loaded_size / total_size * 100:.2f}% of the strategy")
    
    return strategy

@click.command()
@click.option(
    "--low_card_rank",
    default=2,
    help=(
        "The starting hand rank from 2 through 14 for the deck we want to "
        "cluster. We recommend starting small."
    )
)
@click.option(
    "--high_card_rank",
    default=14,
    help=(
        "The starting hand rank from 2 through 14 for the deck we want to "
        "cluster. We recommend starting small."
    )
)
@click.option('--lut_path', required=False, default=".", type=str)
@click.option('--pickle_dir', required=False, default=False, type=bool)
@click.option('--agent', required=False, default="offline", type=str)
@click.option('--strategy_path', required=False, default="", type=str)
@click.option('--previous_strategy_path', required=False, default="", type=str)
@click.option('--debug_quick_start/--no_debug_quick_start', default=False)
def run_terminal_app(
    low_card_rank: int,
    high_card_rank: int,
    lut_path: str,
    pickle_dir: bool,
    agent: str = "offline",
    strategy_path: str = "",
    previous_strategy_path: str = "",
    debug_quick_start: bool = False
):
    """Start up terminal app to play against your poker AI.

    Example
    -------

    Usually you would call this from the `poker_ai` CLI. Alternatively you can
    call this method from this module directly from python.

    ```bash
    python -m poker_ai.terminal.runner                                       \
        --lut_path ./research/blueprint_algo                               \
        --agent offline                                                      \
        --pickle_dir ./research/blueprint_algo                               \
        --strategy_path ./agent.joblib                                       \
        --no_debug_quick_start
    ```
    """
    term = Terminal()
    log = AsciiLogger(term)
    n_players: int = 6
    # n_players: int = 3
    include_ranks = list(range(low_card_rank, high_card_rank + 1))
    if debug_quick_start:
        state: ShortDeckPokerState = new_game(
            n_players,
            {},
            load_card_lut=False,
            include_ranks=include_ranks,
        )
    else:
        state: ShortDeckPokerState = new_game(
            n_players,
            lut_path=lut_path,
            pickle_dir=pickle_dir,
            include_ranks=include_ranks,
        )
    n_table_rotations: int = 0
    selected_action_i: int = 0
    # positions = ["left", "middle", "right"]
    # names = {"left": "BOT 1", "middle": "BOT 2", "right": "HUMAN"}
    positions = ["top-left", "top-middle", "top-right", "bottom-left", "bottom-middle", "bottom-right"]
    names = {"top-left": "BOT 1", "top-middle": "BOT 2", "top-right": "BOT 3", "bottom-left": "BOT 4", "bottom-middle": "BOT 5", "bottom-right": "HUMAN"}
    if not debug_quick_start and agent in {"offline", "online"}:
        print("Pre loading")
        print_memory_usage()
        try:
            offline_strategy_dict = load_strategy(strategy_path)
            print("Strategy loaded successfully")
            print("Keys in loaded data:", offline_strategy_dict.keys())
            
            if 'strategy' in offline_strategy_dict:
                offline_strategy = offline_strategy_dict['strategy']
            else:
                print("'strategy' key not found. Using entire loaded data as strategy.")
                offline_strategy = offline_strategy_dict
            
            # Using the more fine grained preflop strategy would be a good idea
            # for a future improvement
            if 'pre_flop_strategy' in offline_strategy_dict:
                del offline_strategy_dict["pre_flop_strategy"]
            if 'regret' in offline_strategy_dict:
                del offline_strategy_dict["regret"]
            
        except Exception as e:
            print(f"Error loading file: {e}")
        print_memory_usage()
        print("post Loading")

    user_results: UserResults = UserResults()
    with term.cbreak(), term.hidden_cursor():
        while True:
            # Construct ascii objects to be rendered later.
            ascii_players: Dict[str, AsciiPlayer] = {}
            state_players = rotate_list(state.players[::-1], n_table_rotations)
            og_name_to_position = {}
            og_name_to_name = {}
            for player_i, player in enumerate(state_players):
                position = positions[player_i]
                is_human = names[position].lower() == "human"
                ascii_players[position] = AsciiPlayer(
                    *player.cards,
                    term=term,
                    name=names[position],
                    og_name=player.name,
                    hide_cards=not is_human and not state.is_terminal,
                    folded=not player.is_active,
                    is_turn=player.is_turn,
                    chips_in_pot=player.n_bet_chips,
                    chips_in_bank=player.n_chips,
                    is_small_blind=player.is_small_blind,
                    is_big_blind=player.is_big_blind,
                    is_dealer=player.is_dealer,
                )
                og_name_to_position[player.name] = position
                og_name_to_name[player.name] = names[position]
                if player.is_turn:
                    current_player_name = names[position]
            public_cards = AsciiCardCollection(*state.community_cards)
            if state.is_terminal:
                legal_actions = ["quit", "new game"]
                human_should_interact = True
            else:
                og_current_name = state.current_player.name
                human_should_interact = names[og_name_to_position[og_current_name]].lower() == "human"
                if human_should_interact:
                    legal_actions = state.legal_actions
                else:
                    legal_actions = []
            # Render game.
            print(term.home + term.white + term.clear)
            print_header(term, state, og_name_to_name)
            print_table(
                term,
                ascii_players,
                public_cards,
                n_table_rotations,
                n_chips_in_pot=state._table.pot.total,
            )
            print_footer(term, selected_action_i, legal_actions)
            print_log(term, log)
            # Make action of some kind.
            if human_should_interact:
                # Incase the legal_actions went from length 3 to 2 and we had
                # previously picked the last one.
                selected_action_i %= len(legal_actions)
                key = term.inkey(timeout=None)
                if key.name == "q":
                    log.info(term.pink("quit"))
                    break
                elif key.name == "KEY_LEFT":
                    selected_action_i -= 1
                    if selected_action_i < 0:
                        selected_action_i = len(legal_actions) - 1
                elif key.name == "KEY_RIGHT":
                    selected_action_i = (selected_action_i + 1) % len(legal_actions)
                elif key.name == "KEY_ENTER":
                    action = legal_actions[selected_action_i]
                    if action == "quit":
                        user_results.add_result(strategy_path, agent, state, og_name_to_name)
                        log.info(term.pink("quit"))
                        break
                    elif action == "new game":
                        user_results.add_result(strategy_path, agent, state, og_name_to_name)
                        log.clear()
                        log.info(term.green("new game"))
                        include_ranks = list(range(low_card_rank, high_card_rank + 1))
                        if debug_quick_start:
                            state: ShortDeckPokerState = new_game(
                                n_players, state.card_info_lut, load_card_lut=False, include_ranks=include_ranks,
                            )
                        else:
                            state: ShortDeckPokerState = new_game(
                                n_players, state.card_info_lut, include_ranks=include_ranks,
                            )
                        n_table_rotations -= 1
                        if n_table_rotations < 0:
                            n_table_rotations = n_players - 1
                    else:
                        log.info(term.green(f"{current_player_name} chose {action}"))
                        state: ShortDeckPokerState = state.apply_action(action)
            else:
                if agent == "random":
                    action = random.choice(state.legal_actions)
                    time.sleep(0.8)
                elif agent == "offline":
                    default_strategy = {
                        action: 1 / len(state.legal_actions)
                        for action in state.legal_actions
                    }
                    this_state_strategy = offline_strategy.get(
                        state.info_set, default_strategy
                    )
                    # Normalizing strategy.
                    total = sum(this_state_strategy.values())
                    this_state_strategy = {
                        k: v / total for k, v in this_state_strategy.items()
                    }
                    actions = list(this_state_strategy.keys())
                    probabilties = list(this_state_strategy.values())
                    action = np.random.choice(actions, p=probabilties)
                    time.sleep(0.8)
                log.info(f"{current_player_name} chose {action}")
                state: ShortDeckPokerState = state.apply_action(action)


def run_self_play_terminal_app(
    low_card_rank: int,
    high_card_rank: int,
    lut_path: str,
    pickle_dir: bool,
    agent: str = "offline",
    strategy_path: str = "",
    debug_quick_start: bool = False
):
    """Start up terminal app for self-play poker."""
    term = Terminal()
    log = AsciiLogger(term)
    n_players: int = 6
    include_ranks = list(range(low_card_rank, high_card_rank + 1))
    if debug_quick_start:
        state: SelfPlayShortDeckPokerState = new_self_play_game(
            n_players,
            {},
            load_card_lut=False,
            include_ranks=include_ranks,
        )
    else:
        state: SelfPlayShortDeckPokerState = new_self_play_game(
            n_players,
            lut_path=lut_path,
            pickle_dir=pickle_dir,
            include_ranks=include_ranks,
        )
    n_table_rotations: int = 0
    selected_action_i: int = 0
    positions = ["top-left", "top-middle", "top-right", "bottom-left", "bottom-middle", "bottom-right"]
    names = {"top-left": "BOT 1", "top-middle": "BOT 2", "top-right": "BOT 3", "bottom-left": "BOT 4", "bottom-middle": "BOT 5", "bottom-right": "HUMAN"}

    if not debug_quick_start and agent in {"offline", "online"}:
        print("Pre loading")
        print_memory_usage()
        try:
            offline_strategy_dict = load_strategy(strategy_path)
            print("Strategy loaded successfully")
            print("Keys in loaded data:", offline_strategy_dict.keys())
            
            if 'strategy' in offline_strategy_dict:
                offline_strategy = offline_strategy_dict['strategy']
            else:
                print("'strategy' key not found. Using entire loaded data as strategy.")
                offline_strategy = offline_strategy_dict
            
            if 'pre_flop_strategy' in offline_strategy_dict:
                del offline_strategy_dict["pre_flop_strategy"]
            if 'regret' in offline_strategy_dict:
                del offline_strategy_dict["regret"]
            
        except Exception as e:
            print(f"Error loading file: {e}")
        print_memory_usage()
        print("post Loading")

    user_results: UserResults = UserResults()
    with term.cbreak(), term.hidden_cursor():
        while True:
            # Check if we need to deal community cards
            if state.needs_card_input():
                state.input_cards()
                continue

            ascii_players: Dict[str, AsciiPlayer] = {}
            state_players = rotate_list(state.players[::-1], n_table_rotations)
            og_name_to_position = {}
            og_name_to_name = {}
            for player_i, player in enumerate(state_players):
                position = positions[player_i]
                is_human = names[position].lower() == "human"
                ascii_players[position] = AsciiPlayer(
                    *player.cards,
                    term=term,
                    name=names[position],
                    og_name=player.name,
                    hide_cards=not is_human and not state.is_terminal,
                    folded=not player.is_active,
                    is_turn=player.is_turn,
                    chips_in_pot=player.n_bet_chips,
                    chips_in_bank=player.n_chips,
                    is_small_blind=player.is_small_blind,
                    is_big_blind=player.is_big_blind,
                    is_dealer=player.is_dealer,
                )
                og_name_to_position[player.name] = position
                og_name_to_name[player.name] = names[position]
                if player.is_turn:
                    current_player_name = names[position]
            public_cards = AsciiCardCollection(*state.community_cards)
            if state.is_terminal:
                legal_actions = ["quit", "new game"]
                human_should_interact = True
            else:
                og_current_name = state.current_player.name
                human_should_interact = names[og_name_to_position[og_current_name]].lower() == "human"
                if human_should_interact:
                    legal_actions = state.legal_actions
                else:
                    legal_actions = []
            
            print(term.home + term.white + term.clear)
            print_header(term, state, og_name_to_name)
            print_table(
                term,
                ascii_players,
                public_cards,
                n_table_rotations,
                n_chips_in_pot=state._table.pot.total,
            )
            print_footer(term, selected_action_i, legal_actions)
            print_log(term, log)
            
            if human_should_interact:
                selected_action_i %= len(legal_actions)
                key = term.inkey(timeout=None)
                if key.name == "q":
                    log.info(term.pink("quit"))
                    break
                elif key.name == "KEY_LEFT":
                    selected_action_i -= 1
                    if selected_action_i < 0:
                        selected_action_i = len(legal_actions) - 1
                elif key.name == "KEY_RIGHT":
                    selected_action_i = (selected_action_i + 1) % len(legal_actions)
                elif key.name == "KEY_ENTER":
                    action = legal_actions[selected_action_i]
                    if action == "quit":
                        user_results.add_result(strategy_path, agent, state, og_name_to_name)
                        log.info(term.pink("quit"))
                        break
                    elif action == "new game":
                        user_results.add_result(strategy_path, agent, state, og_name_to_name)
                        log.clear()
                        log.info(term.green("new game"))
                        include_ranks = list(range(low_card_rank, high_card_rank + 1))
                        if debug_quick_start:
                            state: SelfPlayShortDeckPokerState = new_self_play_game(
                                n_players, state.card_info_lut, load_card_lut=False, include_ranks=include_ranks,
                            )
                        else:
                            state: SelfPlayShortDeckPokerState = new_self_play_game(
                                n_players, state.card_info_lut, include_ranks=include_ranks,
                            )
                        n_table_rotations -= 1
                        if n_table_rotations < 0:
                            n_table_rotations = n_players - 1
                    else:
                        log.info(term.green(f"{current_player_name} chose {action}"))
                        state: SelfPlayShortDeckPokerState = state.apply_action(action)
            else:
                if agent == "random":
                    action = random.choice(state.legal_actions)
                    time.sleep(0.8)
                elif agent == "offline":
                    default_strategy = {
                        action: 1 / len(state.legal_actions)
                        for action in state.legal_actions
                    }
                    this_state_strategy = offline_strategy.get(
                        state.info_set, default_strategy
                    )
                    total = sum(this_state_strategy.values())
                    this_state_strategy = {
                        k: v / total for k, v in this_state_strategy.items()
                    }
                    actions = list(this_state_strategy.keys())
                    probabilties = list(this_state_strategy.values())
                    action = np.random.choice(actions, p=probabilties)
                    time.sleep(0.8)
                
                log.info(f"{current_player_name} chose {action}")
                state: SelfPlayShortDeckPokerState = state.apply_action(action)

            # After applying the action, check if we need to increment the stage
            if state._poker_engine.n_active_players == 1 or (state.all_players_have_actioned and not state._poker_engine.more_betting_needed):
                state._increment_stage()
                state._reset_betting_round_state()


def select_runner():
    user_input = input("Type '1' if you want to play against the AI, Type '2' if you want the AI to play against a previous version of itself, Type '3' for Self online play: ")
    if user_input == '1':
        run_terminal_app()
    elif user_input == '2':
        run_progress_checker(
            low_card_rank=2,
            high_card_rank=14,
            lut_path="lut://0.0.0.0:8989",
            pickle_dir=False,
            agent="offline",
            strategy_path="./_2024_08_10_01_29_30_879636/agent.joblib",
            previous_strategy_path="./_2024_08_16_05_59_24_726841/agent.joblib",
            debug_quick_start=False
        )
    elif user_input == '3':
        run_self_play_terminal_app(
            low_card_rank=2,
            high_card_rank=14,
            lut_path="lut://0.0.0.0:8989",
            pickle_dir=False,
            agent="offline",
            strategy_path="./_2024_08_10_01_29_30_879636/agent.joblib",
            debug_quick_start=False
        )
    
def run_progress_checker(
    low_card_rank: int,
    high_card_rank: int,
    lut_path: str,
    pickle_dir: bool,
    agent: str = "offline",
    strategy_path: str = "",
    previous_strategy_path: str = "",
    debug_quick_start: bool = False
):
    term = Terminal()
    log = AsciiLogger(term)
    log.height = term.height
    log.info("Running progress checker...")

    # Load the strategies using the load_strategy function
    print("Loading current strategy...")
    print_memory_usage()
    current_strategy_dict = load_strategy(strategy_path)
    print("Current strategy loaded successfully")
    print("Keys in current strategy data:", current_strategy_dict.keys())
    
    if 'strategy' in current_strategy_dict:
        current_strategy = current_strategy_dict['strategy']
    else:
        print("'strategy' key not found in current strategy. Using entire loaded data as strategy.")
        current_strategy = current_strategy_dict
    
    print("Loading previous strategy...")
    print_memory_usage()
    previous_strategy_dict = load_strategy(previous_strategy_path)
    print("Previous strategy loaded successfully")
    print("Keys in previous strategy data:", previous_strategy_dict.keys())
    
    if 'strategy' in previous_strategy_dict:
        previous_strategy = previous_strategy_dict['strategy']
    else:
        print("'strategy' key not found in previous strategy. Using entire loaded data as strategy.")
        previous_strategy = previous_strategy_dict

    print_memory_usage()

    # Load the card_info_lut once
    card_info_lut = ShortDeckPokerState.load_card_lut(
        lut_path=lut_path,
        pickle_dir=pickle_dir,
        low_card_rank=low_card_rank,
        high_card_rank=high_card_rank
    )

    positions = ["top-left", "top-middle", "top-right", "bottom-left", "bottom-middle", "bottom-right"]
    names = {
        "top-left": "Current AI 1", "top-middle": "Previous AI 1", "top-right": "Current AI 2",
        "bottom-left": "Previous AI 2", "bottom-middle": "Current AI 3", "bottom-right": "Previous AI 3"
    }

    n_players = 6
    user_results: UserResults = UserResults()

    total_games = 5000
    games_per_reset = 50
    total_hands = 0
    current_ai_wins = 0
    current_ai_money = 0
    previous_ai_wins = 0
    previous_ai_money = 0

    for reset_count in range(total_games // games_per_reset):
        print(f"\nStarting set {reset_count + 1} of {total_games // games_per_reset}")
        
        for game in range(games_per_reset):
            state: ShortDeckPokerState = new_game(
                n_players=n_players,
                include_ranks=list(range(low_card_rank, high_card_rank + 1)),
                card_info_lut=card_info_lut,
                load_card_lut=False
            )

            hands_this_game = 0
            while not state.is_terminal:
                hands_this_game += 1
                og_current_name = state.current_player.name
                is_current_ai = "Current AI" in names[positions[state.players.index(state.current_player)]]

                strategy = current_strategy if is_current_ai else previous_strategy

                default_strategy = {action: 1 / len(state.legal_actions) for action in state.legal_actions}
                this_state_strategy = strategy.get(state.info_set, default_strategy)
                
                total = sum(this_state_strategy.values())
                this_state_strategy = {k: v / total for k, v in this_state_strategy.items()}
                
                actions = list(this_state_strategy.keys())
                probabilities = list(this_state_strategy.values())
                action = np.random.choice(actions, p=probabilities)
                
                state = state.apply_action(action)

            # Game ended
            total_hands += hands_this_game
            
            # Update statistics
            for player in state.players:
                if "Current AI" in names[positions[state.players.index(player)]]:
                    if player.n_chips > 0:
                        current_ai_wins += 1
                    current_ai_money += player.n_chips - 10000  # Assuming starting chips is 10000
                else:
                    if player.n_chips > 0:
                        previous_ai_wins += 1
                    previous_ai_money += player.n_chips - 10000  # Assuming starting chips is 10000  # Assuming starting chips is 10000

        # Print progress after each reset
        games_played = (reset_count + 1) * games_per_reset
        print(f"\nProgress after {games_played} games:")
        print(f"Total hands played: {total_hands}")
        ###print(f"Current AI wins: {current_ai_wins}")
        ###print(f"Previous AI wins: {previous_ai_wins}")
        ###print(f"Current AI total money won/lost: ${current_ai_money}")
        ###print(f"Previous AI total money won/lost: ${previous_ai_money}")
        ###print(f"Current AI average money per game: ${current_ai_money / games_played:.2f}")
        ###print(f"Previous AI average money per game: ${previous_ai_money / (games_played * 5):.2f}")

    print("\nFinal Results after 5000 games:")
    print(f"Total hands played: {total_hands}")
    print(f"Current AI wins: {current_ai_wins}")
    print(f"Previous AI wins: {previous_ai_wins}")
    print(f"Current AI total money won/lost: ${current_ai_money}")
    print(f"Previous AI total money won/lost: ${previous_ai_money}")
    print(f"Current AI average money per game: ${current_ai_money / total_games:.2f}")
    print(f"Previous AI average money per game: ${previous_ai_money / (total_games * 5):.2f}")

    log.info("Finished comparing strategies.")
        

    print("Final Results:")

if __name__ == "__main__":
    select_runner()
