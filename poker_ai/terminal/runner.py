import random
import time
from typing import Dict

import click
import joblib
import numpy as np
from blessed import Terminal

from poker_ai.games.short_deck.state import new_game, ShortDeckPokerState
from poker_ai.terminal.ascii_objects.card_collection import AsciiCardCollection
from poker_ai.terminal.ascii_objects.player import AsciiPlayer
from poker_ai.terminal.ascii_objects.logger import AsciiLogger
from poker_ai.terminal.render import print_footer, print_header, print_log, print_table
from poker_ai.terminal.results import UserResults
from poker_ai.utils.algos import rotate_list


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
        offline_strategy_dict = joblib.load(strategy_path)
        offline_strategy = offline_strategy_dict['strategy']
        # Using the more fine grained preflop strategy would be a good idea
        # for a future improvement
        del offline_strategy_dict["pre_flop_strategy"]
        del offline_strategy_dict["regret"]
    else:
        offline_strategy = {}
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


def select_runner():
    user_input = input("Type '1' if you want to play against the AI, Type '2' if you want the AI to play against a previous version of itself: ")
    if user_input == '1':
        run_terminal_app()
    elif user_input == '2':
        run_progress_checker(
            low_card_rank=4,
            high_card_rank=11,
            lut_path=".",
            pickle_dir=False,
            agent="offline",
            strategy_path="./_2024_07_10_19_16_31_142882/agent.joblib",
            previous_strategy_path="./_2024_07_10_19_16_31_142882/agent.joblib",
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

    # Load the strategies only once at the beginning
    current_strategy_dict = joblib.load(strategy_path)
    current_strategy = current_strategy_dict['strategy']
    previous_strategy_dict = joblib.load(previous_strategy_path)
    previous_strategy = previous_strategy_dict['strategy']

    # Load the card_info_lut once
    card_info_lut = ShortDeckPokerState.load_card_lut(
        lut_path=lut_path,
        pickle_dir=pickle_dir,
        low_card_rank=low_card_rank,
        high_card_rank=high_card_rank
    )

    positions = ["top-left", "top-middle", "top-right", "bottom-left", "bottom-middle", "bottom-right"]
    names = {"top-left": "Previous AI", "top-middle": "Previous AI", "top-right": "Previous AI", 
             "bottom-left": "Previous AI", "bottom-middle": "Previous AI", "bottom-right": "Current AI"}

    n_players = 6
    user_results: UserResults = UserResults()

    while True:
        games_played = 0
        max_games = 50
        total_hands = 0
        current_ai_wins = 0
        current_ai_money = 0
        previous_ai_wins = 0
        previous_ai_money = 0

        while games_played < max_games:
            state: ShortDeckPokerState = new_game(
                n_players=n_players,
                include_ranks=list(range(low_card_rank, high_card_rank + 1)),
                card_info_lut=card_info_lut,  # Pass the pre-loaded card_info_lut
                load_card_lut=False  # Set this to False to avoid reloading
            )

            hands_this_game = 0
            while not state.is_terminal:
                hands_this_game += 1
                og_current_name = state.current_player.name
                is_current_ai = names[positions[state.players.index(state.current_player)]] == "Current AI"

                strategy = current_strategy if is_current_ai else previous_strategy

                default_strategy = {action: 1 / len(state.legal_actions) for action in state.legal_actions}
                this_state_strategy = strategy.get(state.info_set, default_strategy)
                
                total = sum(this_state_strategy.values())
                this_state_strategy = {k: v / total for k, v in this_state_strategy.items()}
                
                actions = list(this_state_strategy.keys())
                probabilities = list(this_state_strategy.values())
                action = np.random.choice(actions, p=probabilities)
                
                log.info(f"{names[positions[state.players.index(state.current_player)]]} chose {action}")
                state = state.apply_action(action)

            # Game ended
            total_hands += hands_this_game
            
            # Update statistics
            for player in state.players:
                if names[positions[state.players.index(player)]] == "Current AI":
                    if player.n_chips > 0:
                        current_ai_wins += 1
                    current_ai_money += player.n_chips - 10000  # Assuming starting chips is 10000
                else:
                    if player.n_chips > 0:
                        previous_ai_wins += 1
                    previous_ai_money += player.n_chips - 10000  # Assuming starting chips is 10000

            games_played += 1

        print(term.home + term.white + term.clear)
        print_log(term, log)
        
        print("\nStatistics after 5 games:")
        print(f"Total hands played: {total_hands}")
        print(f"Current AI wins: {current_ai_wins}")
        print(f"Previous AI wins: {previous_ai_wins}")
        print(f"Current AI total money won/lost: ${current_ai_money}")
        print(f"Previous AI total money won/lost: ${previous_ai_money}")
        print(f"Current AI average money per game: ${current_ai_money / max_games:.2f}")
        print(f"Previous AI average money per game: ${previous_ai_money / (max_games * 5):.2f}")
        
        user_input = input("50 games completed. Enter 'q' to quit or any other key to continue: ")
        if user_input.lower() == 'q':
            break

    log.info("Finished comparing strategies.")
    print("Final Results:")
    print(user_results.get_summary())

if __name__ == "__main__":
    select_runner()
