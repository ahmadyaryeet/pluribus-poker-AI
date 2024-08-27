from __future__ import annotations

from typing import List, TYPE_CHECKING

from poker_ai.poker.deck import Deck

if TYPE_CHECKING:
    from poker_ai.poker.table import PokerTable
    from poker_ai.poker.player import Player
    from poker_ai.poker.card import Card


class Dealer:
    """The dealer is in charge of handling the cards on a poker table."""

    def __init__(self, **deck_kwargs):
        self.deck = Deck(**deck_kwargs)

    def deal_card(self) -> Card:
        """Return a completely random card."""
        return self.deck.pick(random=True)

    def deal_private_cards(self, players: List[Player]):
        """Deal private card to players."""
        for _ in range(2):
            for player in players:
                card: Card = self.deal_card()
                player.add_private_card(card)

    def deal_community_cards(self, table: PokerTable, n_cards: int):
        """Deal public cards."""
        if n_cards <= 0:
            raise ValueError(
                f"Positive n of cards must be specified, but got {n_cards}"
            )
        for _ in range(n_cards):
            card: Card = self.deal_card()
            table.add_community_card(card)

    def deal_flop(self, table: PokerTable):
        """Deal the flop public cards to the `table`."""
        return self.deal_community_cards(table, 3)

    def deal_turn(self, table: PokerTable):
        """Deal the turn public cards to the `table`."""
        return self.deal_community_cards(table, 1)

    def deal_river(self, table: PokerTable):
        """Deal the river public cards to the `table`."""
        return self.deal_community_cards(table, 1)

    # Self-play duplicate functions
    def self_play_get_user_input_cards(self, stage: str, num_cards: int) -> List[Card]:
        """
        Ask the user to input specific cards for a given stage.
        
        Parameters:
        - stage: str, the current stage of the game (e.g., "flop", "turn", "river", "player hand")
        - num_cards: int, the number of cards to request from the user
        
        Returns:
        - List[Card]: A list of Card objects based on user input
        """
        print(f"Enter the cards for the {stage} (format: rank suit, e.g., 'AS KC QH'):")
        while True:
            try:
                user_input = input().upper().split()
                if len(user_input) != num_cards:
                    raise ValueError(f"Please enter exactly {num_cards} card(s).")
                
                cards = []
                for card_str in user_input:
                    if len(card_str) != 2:
                        raise ValueError(f"Invalid card format: {card_str}. Use format like 'AS' for Ace of Spades.")
                    rank, suit = card_str[0], card_str[1]
                    card = Card.from_string(f"{rank}{suit}")
                    cards.append(card)
                
                return cards
            except ValueError as e:
                print(f"Error: {e} Please try again.")

    def self_play_deal_private_cards(self, players: List[Player]):
        """Deal private cards to players for self-play."""
        for i, player in enumerate(players):
            cards = self.self_play_get_user_input_cards(f"Player {i+1}'s hand", 2)
            for card in cards:
                player.add_private_card(card)

    def self_play_deal_flop(self, table: PokerTable):
        """Deal the flop public cards to the `table` for self-play."""
        cards = self.self_play_get_user_input_cards("flop", 3)
        for card in cards:
            table.add_community_card(card)

    def self_play_deal_turn(self, table: PokerTable):
        """Deal the turn public card to the `table` for self-play."""
        cards = self.self_play_get_user_input_cards("turn", 1)
        table.add_community_card(cards[0])

    def self_play_deal_river(self, table: PokerTable):
        """Deal the river public card to the `table` for self-play."""
        cards = self.self_play_get_user_input_cards("river", 1)
        table.add_community_card(cards[0])

    def self_play_deal_community_cards(self, table: PokerTable, n_cards: int):
        """Deal public cards for self-play."""
        if n_cards <= 0:
            raise ValueError(f"Positive n of cards must be specified, but got {n_cards}")
        
        stage = "community cards"
        if n_cards == 3:
            stage = "flop"
        elif n_cards == 1:
            stage = "turn" if len(table.community_cards) == 3 else "river"
        
        cards = self.self_play_get_user_input_cards(stage, n_cards)
        for card in cards:
            table.add_community_card(card)
