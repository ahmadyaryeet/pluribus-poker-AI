from __future__ import annotations

import collections
import copy
import json
import logging
import operator
import os
from typing import Any, Dict, List, Optional, Tuple

import pickle

from poker_ai import utils
from poker_ai.clustering.lookup_client import LookupClient
from poker_ai.poker.card import Card
from poker_ai.poker.engine import PokerEngine
from poker_ai.games.short_deck.player import ShortDeckPokerPlayer
from poker_ai.poker.pot import Pot
from poker_ai.poker.table import PokerTable

logger = logging.getLogger("poker_ai.games.short_deck.self_play_state")
InfoSetLookupTable = Dict[str, Dict[Tuple[int, ...], str]]
raise_levels = [2, 5, 50]

def new_self_play_game(
    n_players: int,
    card_info_lut: InfoSetLookupTable = None,
    **kwargs
) -> SelfPlayShortDeckPokerState:
    pot = Pot()
    players = [
        ShortDeckPokerPlayer(player_i=player_i, initial_chips=10000, pot=pot)
        for player_i in range(n_players)
    ]
    if card_info_lut:
        state = SelfPlayShortDeckPokerState(
            players=players,
            card_info_lut=card_info_lut,
            **kwargs
        )
    else:
        state = SelfPlayShortDeckPokerState(
            players=players,
            card_info_lut=card_info_lut,
            **kwargs
        )
    return state

class SelfPlayShortDeckPokerState:
    def __init__(
        self,
        players: List[ShortDeckPokerPlayer],
        small_blind: int = 50,
        big_blind: int = 100,
        lut_path: str = ".",
        pickle_dir: bool = False,
        load_card_lut: bool = True,
        include_ranks: List[int] = None,
        card_info_lut: Optional[Dict[str, Any]] = None,
        without_blinds: bool = False,
        handle_all_in: bool = False,
        allow_fourth_bet: bool = False,
    ):
        n_players = len(players)
        if n_players <= 1:
            raise ValueError(
                f"At least 2 players must be provided but only {n_players} "
                f"were provided."
            )
        self.handle_all_in = handle_all_in
        self.allow_fourth_bet = allow_fourth_bet
        self._pickle_dir = pickle_dir
        
        self._low_card_rank = min(include_ranks)
        self._high_card_rank = max(include_ranks)
        if card_info_lut is not None:
            self.card_info_lut = card_info_lut
        elif load_card_lut:
            self.card_info_lut = self.load_card_lut(
                lut_path, self._pickle_dir, self._low_card_rank, self._high_card_rank
            )
        else:
            logger.warning("Initializing a SelfPlayPokerState without a lookup table.")
            self.card_info_lut = {}
        
        self._table = PokerTable(
            players=players, pot=players[0].pot, include_ranks=include_ranks,
        )
        self._initial_n_chips = players[0].n_chips
        self.small_blind = small_blind
        self.big_blind = big_blind
        self._poker_engine = PokerEngine(
            table=self._table,
            small_blind=small_blind,
            big_blind=big_blind,
            handle_all_in=self.handle_all_in,
        )
        
        self._poker_engine.round_setup(without_blinds=without_blinds)
        self._table.dealer.self_play_deal_private_cards(self._table.players)
        
        self._history: Dict[str, List[str]] = collections.defaultdict(list)
        self._betting_stage = "pre_flop"
        self._betting_stage_to_round: Dict[str, int] = {
            "pre_flop": 0,
            "flop": 1,
            "turn": 2,
            "river": 3,
            "show_down": 4,
        }
        
        player_i_order: List[int] = [p_i for p_i in range(n_players)]
        self.players[-1].is_dealer = True
        self._player_i_lut: Dict[str, List[int]] = {
            "pre_flop": player_i_order[2:] + player_i_order[:2],
            "flop": player_i_order,
            "turn": player_i_order,
            "river": player_i_order,
            "show_down": player_i_order,
            "terminal": player_i_order,
        }
        self._skip_counter = 0
        self._first_move_of_current_round = True
        self._reset_betting_round_state()
        for player in self.players:
            player.is_turn = False
        self.current_player.is_turn = True

    def smart_assign_blinds(self):
        blind_players = self._poker_engine.smart_assign_blinds()
        blind_players[0].is_small_blind = True
        blind_players[1].is_big_blind = True
    
    def input_cards(self):
        if self._betting_stage == "pre_flop":
            for i, player in enumerate(self.players):
                if not player.cards:
                    cards = self._table.dealer.self_play_get_user_input_cards(f"Player {i+1}'s hand", 2)
                    player.cards = cards
        elif self._betting_stage == "flop":
            cards = self._table.dealer.self_play_get_user_input_cards("flop", 3)
            self._table.community_cards = cards
        elif self._betting_stage == "turn":
            cards = self._table.dealer.self_play_get_user_input_cards("turn", 1)
            self._table.community_cards.extend(cards)
        elif self._betting_stage == "river":
            cards = self._table.dealer.self_play_get_user_input_cards("river", 1)
            self._table.community_cards.extend(cards)

    def needs_card_input(self) -> bool:
        if self._betting_stage == "pre_flop" and not all(player.cards for player in self.players):
            return True
        if self._betting_stage == "flop" and len(self._table.community_cards) < 3:
            return True
        if self._betting_stage == "turn" and len(self._table.community_cards) == 3:
            return True
        if self._betting_stage == "river" and len(self._table.community_cards) == 4:
            return True
        return False

    def skip_players_with_no_chips(self):
        while self.current_player.n_chips == 0:
            self.current_player.is_active = False
            self._skip_counter += 1
            self._move_to_next_player()
    
    def flag_broke_players(self):
        for player in self.players:
            if player.n_chips == 0:
                player.is_broke = True

    def __repr__(self):
        return f"<SelfPlayShortDeckPokerState player_i={self.player_i} betting_stage={self._betting_stage}>"

    # ... (include all other methods from the original ShortDeckPokerState class)

    def apply_action(self, action_str: Optional[str]) -> SelfPlayShortDeckPokerState:
        # ... (same as in ShortDeckPokerState, but return SelfPlayShortDeckPokerState)
        """Create a new state after applying an action.

        Parameters
        ----------
        action_str : str or None
            The description of the action the current player is making. Can be
            any of {"fold, "call", "raise"}, the latter two only being possible
            if the agent hasn't folded already.

        Returns
        -------
        new_state : ShortDeckPokerState
            A poker state instance that represents the game in the next
            timestep, after the action has been applied.
        """
        if action_str.split(":")[0] not in self.legal_actions:
            raise ValueError(
                f"Action '{action_str}' not in legal actions: " f"{self.legal_actions}"
            )
        # Deep copy the parts of state that are needed that must be immutable
        # from state to state.
        lut = self.card_info_lut
        self.card_info_lut = {}
        new_state = copy.deepcopy(self)
        new_state.card_info_lut = self.card_info_lut = lut
        # An action has been made, so alas we are not in the first move of the
        # current betting round.
        new_state._first_move_of_current_round = False
        raise_action = None
        if action_str is None:
            # Assert active player has folded already.
            assert (
                not new_state.current_player.is_active
            ), "Active player cannot do nothing!"
        elif action_str == "call":
            action = new_state.current_player.call(players=new_state.players)
            logger.debug("calling")
        elif action_str == "fold":
            action = new_state.current_player.fold()
        elif action_str.startswith("raise"):
            bet_n_chips = new_state.big_blind
            if new_state._betting_stage in {"turn", "river"}:
                bet_n_chips *= 2
            if len(action_str.split(":")) > 1:
                param = action_str.split(":")[-1]
                if param.startswith("lv"):
                    level = int(param.split("lv")[1])
                    bet_n_chips *= level
                    raise_action = action_str
                else:
                    base_bet_n_chips = bet_n_chips
                    custom_bet_n_chips = int(param)
                    if custom_bet_n_chips > bet_n_chips:
                        bet_n_chips = custom_bet_n_chips
                    approximate_level = 1
                    for level in raise_levels:
                        if bet_n_chips >= base_bet_n_chips * level:
                            approximate_level = level
                    if approximate_level > 1:
                        raise_action = f"raise:lv{approximate_level}"
            biggest_bet = max(p.n_bet_chips for p in new_state.players)
            n_chips_to_call = biggest_bet - new_state.current_player.n_bet_chips
            raise_n_chips = bet_n_chips + n_chips_to_call
            logger.debug(f"betting {raise_n_chips} n chips")
            action = new_state.current_player.raise_to(n_chips=raise_n_chips)
            new_state._n_raises += 1
        else:
            raise ValueError(
                f"Expected action to be derived from class Action, but found "
                f"type {type(action)}."
            )
        # Update the new state.
        skip_actions = ["skip" for _ in range(new_state._skip_counter)]
        new_state._history[new_state.betting_stage] += skip_actions
        if raise_action is not None:
            new_state._history[new_state.betting_stage].append(raise_action)
        else:
            new_state._history[new_state.betting_stage].append(str(action))
        new_state._n_actions += 1
        new_state._skip_counter = 0
        # Player has made move, increment the player that is next.
        while True:
            new_state._move_to_next_player()
            # If we have finished betting, (i.e: All players have put the
            # same amount of chips in), then increment the stage of
            # betting.
            finished_betting = not new_state._poker_engine.more_betting_needed
            if finished_betting and new_state.all_players_have_actioned:
                # We have done atleast one full round of betting, increment
                # stage of the game.
                new_state._increment_stage()
                new_state._reset_betting_round_state()
                new_state._first_move_of_current_round = True
            if not new_state.current_player.is_active:
                new_state._skip_counter += 1
                assert not new_state.current_player.is_active
            elif new_state.current_player.is_broke and self.handle_all_in:
                # Auto-fold broke players.
                new_state.current_player.is_active = False
                new_state._skip_counter += 1
            elif new_state.current_player.is_active:
                # if new_state._poker_engine.n_players_with_moves == 1:
                if new_state._poker_engine.n_active_players == 1:
                    # No players left.
                    new_state._betting_stage = "terminal"
                    if not new_state._table.community_cards:
                        new_state._poker_engine.table.dealer.deal_flop(new_state._table)
                # Now check if the game is terminal.
                if new_state._betting_stage in {"terminal", "show_down"}:
                    # Distribute winnings.
                    new_state._poker_engine.compute_winners()
                break
        for player in new_state.players:
            player.is_turn = False
        new_state.current_player.is_turn = True
        return new_state

    @staticmethod
    def load_card_lut(
        lut_path: str = ".",
        pickle_dir: bool = False,
        #editing these values from static values to being from 2 to 7, and 14 to 11.

        low_card_rank: int = 2,
        high_card_rank: int = 14,
    ) -> Dict[str, Dict[Tuple[int, ...], str]]:
        """
        Load card information lookup table.

        ...

        Parameters
        ----------
        lut_path : str
            Path to lookup table.
        pickle_dir : bool
            Whether the lut_path is a path to pickle files or not. Pickle files
            are deprecated for the lut.

        Returns
        -------
        cad_info_lut : InfoSetLookupTable
            Card information cluster lookup table.
        """
        if pickle_dir:
            logger.info("Loading card information lut in deprecated way")
            file_names = [
                "preflop_lossless.pkl",
                "flop_lossy_2.pkl",
                "turn_lossy_2.pkl",
                "river_lossy_2.pkl",
            ]
            betting_stages = ["pre_flop", "flop", "turn", "river"]
            card_info_lut: Dict[str, Dict[Tuple[int, ...], str]] = {}
            for file_name, betting_stage in zip(file_names, betting_stages):
                file_path = os.path.join(lut_path, file_name)
                if not os.path.isfile(file_path):
                    raise ValueError(
                        f"File path not found {file_path}. Ensure lut_path is "
                        f"set to directory containing pickle files"
                    )
                with open(file_path, "rb") as fp:
                    card_info_lut[betting_stage] = pickle.load(fp)
        elif lut_path and lut_path.startswith("lut://"):
            logger.info(f"Connecting to a card information lut server: {lut_path}")
            card_info_lut = LookupClient(
                lut_path,
                low_card_rank=low_card_rank,
                high_card_rank=high_card_rank,
            )
            card_info_lut.connect()
        elif lut_path:
            logger.info(f"Loading card from single file at path: {lut_path}")
            #filename = f"card_info_lut_{low_card_rank}_to_{high_card_rank}.joblib"
            filename = f"card_info_lut_{low_card_rank}_to_{high_card_rank}.joblib"
            with open(lut_path + "/" + filename, 'rb') as f:
                card_info_lut = pickle.load(f)
        else:
            card_info_lut = {}
        return card_info_lut
    
    @property
    def community_cards(self) -> List[Card]:
        return self._table.community_cards
    
    @property
    def private_hands(self) -> Dict[ShortDeckPokerPlayer, List[Card]]:
        return {p: p.cards for p in self.players}

    @property
    def initial_regret(self) -> Dict[str, float]:
        return {action: 0 for action in self.legal_traverse_actions}

    @property
    def initial_strategy(self) -> Dict[str, float]:
        return {action: 0 for action in self.legal_traverse_actions}

    @property
    def betting_stage(self) -> str:
        return self._betting_stage

    @property
    def all_players_have_actioned(self) -> bool:
        return self._n_actions >= self._n_players_started_round

    @property
    def n_players_started_round(self) -> bool:
        return self._n_players_started_round

    @property
    def player_i(self) -> int:
        return self._player_i_lut[self._betting_stage][self._player_i_index]

    @player_i.setter
    def player_i(self, _: Any):
        raise ValueError(f"The player_i property should not be set.")

    @property
    def betting_round(self) -> int:
        try:
            betting_round = self._betting_stage_to_round[self._betting_stage]
        except KeyError:
            raise ValueError(
                f"Attemped to get betting round for stage "
                f"{self._betting_stage} but was not supported in the lut with "
                f"keys: {list(self._betting_stage_to_round.keys())}"
            )
        return betting_round

    @property
    def info_set(self) -> str:
        """Get the information set for the current player."""
        cards = sorted(
            self.current_player.cards,
            key=operator.attrgetter("eval_card"),
            reverse=True,
        )
        cards += sorted(
            self._table.community_cards,
            key=operator.attrgetter("eval_card"),
            reverse=True,
        )
        
        lookup_cards = tuple([card.eval_card for card in cards])
        # cards_cluster = 1
        if self.card_info_lut != {}:
            try:
                cards_cluster = self.card_info_lut[self._betting_stage][lookup_cards]
            except KeyError:
                if self.betting_stage not in {"terminal", "show_down"}:
                    raise ValueError("You should have these cards in your lut.")
                return "default info set, please ensure you load it correctly"
        else:
            cards_cluster = 1

        # Convert history from a dict of lists to a list of dicts as I'm
        # paranoid about JSON's lack of care with insertion order.
        info_set_dict = {
            "cards_cluster": cards_cluster,
            "history": [
                {betting_stage: [str(action) for action in actions]}
                for betting_stage, actions in self._history.items()
            ],
        }
        return json.dumps(
            info_set_dict, separators=(",", ":"), cls=utils.io.NumpyJSONEncoder
        )

    @property
    def payout(self) -> Dict[int, int]:
        """Return player index to payout number of chips dictionary."""
        n_chips_delta = dict()
        for player_i, player in enumerate(self.players):
            n_chips_delta[player_i] = player.n_chips - self._initial_n_chips
        return n_chips_delta

    @property
    def is_terminal(self) -> bool:
        """Returns whether this state is terminal or not.

        The state is terminal once all rounds of betting are complete and we
        are at the show down stage of the game or if all players have folded.
        """
        return self._betting_stage in {"show_down", "terminal"}

    @property
    def players(self) -> List[ShortDeckPokerPlayer]:
        """Returns players in table."""
        return self._table.players

    @property
    def current_player(self) -> ShortDeckPokerPlayer:
        """Returns a reference to player that makes a move for this state."""
        return self._table.players[self.player_i]
    
    @property
    def raise_limit(self) -> int:
        return 4 if self.allow_fourth_bet else 3

    @property
    def legal_actions(self) -> List[Optional[str]]:
        """Return the actions that are legal for this game state."""
        actions: List[Optional[str]] = []
        if self.current_player.is_active:
            actions += ["fold", "call"]
            if self._n_raises < self.raise_limit:
                # In limit hold'em we can only bet/raise if there have been
                # less than three or four raises in this round of betting, or if there
                # are two players playing.
                actions += ["raise"]
        else:
            actions += [None]
        return actions
    
    @property
    def legal_traverse_actions(self) -> List[Optional[str]]:
        """Return the actions that are legal for this game state for traverse.
        
        Return the available actions including custom amount raises.
        """
        actions: List[Optional[str]] = []
        if self.current_player.is_active:
            actions.append("fold")
            actions.append("call")
            if self._n_raises < self.raise_limit:
                # In limit hold'em we can only bet/raise if there have been
                # less than three raises in this round of betting, or if there
                # are two players playing.
                actions.append("raise")
                for level in raise_levels:
                    actions.append(f"raise:lv{level}")
        else:
            actions.append(None)
        return actions

    # ... (include all other methods and properties from ShortDeckPokerState)

    def set_community_cards(self, cards: List[Card]):
        if self._betting_stage == "flop":
            if len(cards) != 3:
                raise ValueError("Flop must have exactly 3 cards.")
            self._table.dealer.self_play_deal_flop(self._table)
        elif self._betting_stage == "turn":
            if len(cards) != 1:
                raise ValueError("Turn must have exactly 1 card.")
            self._table.dealer.self_play_deal_turn(self._table)
        elif self._betting_stage == "river":
            if len(cards) != 1:
                raise ValueError("River must have exactly 1 card.")
            self._table.dealer.self_play_deal_river(self._table)
        else:
            raise ValueError(f"Cannot set community cards in {self._betting_stage} stage.")

    def set_player_cards(self, player_index: int, cards: List[Card]):
        if len(cards) != 2:
            raise ValueError("Each player must have exactly 2 cards.")
        self.players[player_index].cards = cards

    def _increment_stage(self):
        """Once betting has finished, increment the stage of the poker game."""
        if self._betting_stage == "pre_flop":
            self._betting_stage = "flop"
            # Remove the automatic dealing of flop cards
            # self._table.dealer.self_play_deal_flop(self._table)
        elif self._betting_stage == "flop":
            self._betting_stage = "turn"
            # Remove the automatic dealing of turn card
            # self._table.dealer.self_play_deal_turn(self._table)
        elif self._betting_stage == "turn":
            self._betting_stage = "river"
            # Remove the automatic dealing of river card
            # self._table.dealer.self_play_deal_river(self._table)
        elif self._betting_stage == "river":
            self._betting_stage = "show_down"
        elif self._betting_stage in {"show_down", "terminal"}:
            pass
        else:
            raise ValueError(f"Unknown betting_stage: {self._betting_stage}")
        
    def _move_to_next_player(self):
        """Ensure state points to next valid active player."""
        self._player_i_index += 1
        if self._player_i_index >= len(self.players):
            self._player_i_index = 0
        
    def _reset_betting_round_state(self):
        """Reset the state related to counting types of actions."""
        self._all_players_have_made_action = False
        self._n_actions = 0
        self._n_raises = 0
        self._player_i_index = 0
        self._n_players_started_round = self._poker_engine.n_active_players
        while not self.current_player.is_active:
            self._skip_counter += 1
            self._player_i_index += 1

    def set_player_cards(self, player_index: int, cards: List[Card]):
        """Set the private cards for a specific player."""
        if len(cards) != 2:
            raise ValueError("Each player must have exactly 2 cards.")
        self.players[player_index].cards = cards

    # Add any other methods specific to self-play here