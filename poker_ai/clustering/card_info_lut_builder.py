import logging
import time
import math
from pathlib import Path
from typing import Any, Dict, Optional
import pickle
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from scipy.stats import wasserstein_distance
from tqdm import tqdm
from multiprocessing import Pool, cpu_count, shared_memory

from poker_ai.clustering.card_combos import CardCombos
from poker_ai.clustering.combo_lookup import ComboLookup
from poker_ai.clustering.game_utility import GameUtility
from poker_ai.clustering.preflop import compute_preflop_lossy_abstraction
from poker_ai.poker.evaluation import Evaluator
from poker_ai.utils.safethread import multiprocess_ehs_calc

log = logging.getLogger("poker_ai.clustering.runner")

def process_river_ehs(public: np.ndarray, evaluator: Evaluator, n_simulations_river: int) -> np.ndarray:
        available_cards = np.array([c for c in evaluator._cards if c not in public])
        prob_unit = 1 / n_simulations_river
        ehs: np.ndarray = np.zeros(3)
        opp_hand = public.copy()
        for _ in range(n_simulations_river):
            opp_hand[:2] = np.random.choice(available_cards, 2, replace=False)
            opp_hand_rank = evaluator._seven(opp_hand)
            if evaluator._seven(public) > opp_hand_rank:
                ehs[0] += prob_unit
            elif evaluator._seven(public) < opp_hand_rank:
                ehs[1] += prob_unit
            else:
                ehs[2] += prob_unit
        return ehs


class CardInfoLutBuilder(CardCombos):
    def __init__(
        self,
        n_simulations_river: int,
        n_simulations_turn: int,
        n_simulations_flop: int,
        low_card_rank: int,
        high_card_rank: int,
        save_dir: str,
    ):
        self._evaluator = Evaluator()
        self.n_simulations_river = n_simulations_river
        self.n_simulations_turn = n_simulations_turn
        self.n_simulations_flop = n_simulations_flop
        super().__init__(low_card_rank, high_card_rank, save_dir)
        card_info_lut_filename = f"card_info_lut_{low_card_rank}_to_{high_card_rank}.joblib"
        centroid_filename = f"centroids_{low_card_rank}_to_{high_card_rank}.joblib"
        self.card_info_lut_path: Path = Path(save_dir) / card_info_lut_filename
        self.centroid_path: Path = Path(save_dir) / centroid_filename

        try:
            with open(self.card_info_lut_path, 'rb') as f:
                self.card_info_lut: Dict[str, Any] = pickle.load(f)
            with open(self.centroid_path, 'rb') as f:
                self.centroids: Dict[str, Any] = pickle.load(f)
        except FileNotFoundError:
            self.centroids: Dict[str, Any] = {}
            self.card_info_lut: Dict[str, Any] = {}

    def load_raw_card_lookup(self, combos_path, clusters_path, line_count):
        with open(combos_path, "r") as combos_file, open(clusters_path, "r") as clusters_file:
            lossy_lookup = ComboLookup()
            with tqdm(total=line_count, ascii=" >=") as pbar:
                while True:
                    combos_line = combos_file.readline().strip()
                    clusters_line = clusters_file.readline().strip()
                    if not combos_line or not clusters_line:
                        break

                    cluster = int(clusters_line)
                    combo = [int(x) for x in combos_line.split(",")]
                    lossy_lookup[combo] = cluster
                    pbar.update(1)
        return lossy_lookup

    def load_raw_centroids(self, centroids_path):
        centroids = []
        with open(centroids_path, "r") as f:
            for line in f:
                centroid = [float(x) for x in line.split(",")]
                centroids.append(centroid)
        return centroids

    def load_raw_dir(self, raw_dir: str):
        log.info("Calculating pre-flop abstraction.")
        self.card_info_lut["pre_flop"] = compute_preflop_lossy_abstraction(
            builder=self
        )

        raw_dir_path = Path(raw_dir)
        deck_size = len(self._cards)

        raw_river_combos_path: Path = raw_dir_path / "river_combos.txt"
        raw_river_clusters_path: Path = raw_dir_path / "river_clusters.txt"
        raw_river_centroids_path: Path = raw_dir_path / "river_centroids.txt"
        river_size = math.comb(deck_size, 2) * math.comb(deck_size - 2, 5)

        log.info("Creating river lookup table.")
        self.card_info_lut["river"] = self.load_raw_card_lookup(
            raw_river_combos_path,
            raw_river_clusters_path,
            river_size,
        )
        self.centroids["river"] = self.load_raw_centroids(raw_river_centroids_path)

        raw_turn_combos_path: Path = raw_dir_path / "turn_combos.txt"
        raw_turn_clusters_path: Path = raw_dir_path / "turn_clusters.txt"
        raw_turn_centroids_path: Path = raw_dir_path / "turn_centroids.txt"
        turn_size = math.comb(deck_size, 2) * math.comb(deck_size - 2, 4)

        log.info("Creating turn lookup table.")
        self.card_info_lut["turn"] = self.load_raw_card_lookup(
            raw_turn_combos_path,
            raw_turn_clusters_path,
            turn_size,
        )
        self.centroids["turn"] = self.load_raw_centroids(raw_turn_centroids_path)

        raw_flop_combos_path: Path = raw_dir_path / "flop_combos.txt"
        raw_flop_clusters_path: Path = raw_dir_path / "flop_clusters.txt"
        raw_flop_centroids_path: Path = raw_dir_path / "flop_centroids.txt"
        flop_size = math.comb(deck_size, 2) * math.comb(deck_size - 2, 3)

        log.info("Creating flop lookup table.")
        self.card_info_lut["flop"] = self.load_raw_card_lookup(
            raw_flop_combos_path,
            raw_flop_clusters_path,
            flop_size,
        )
        self.centroids["flop"] = self.load_raw_centroids(raw_flop_centroids_path)

        with open(self.card_info_lut_path, 'wb') as f:
            pickle.dump(self.card_info_lut, f)
        with open(self.centroid_path, 'wb') as f:
            pickle.dump(self.centroids, f)

    def compute(
        self, n_river_clusters: int, n_turn_clusters: int, n_flop_clusters: int,
    ):
        """Compute all clusters and save to card_info_lut dictionary.

        Will attempt to load previous progress and will save after each cluster
        is computed.
        """
        log.info("Starting computation of clusters.")
        start = time.time()
        if "pre_flop" not in self.card_info_lut:
            self.card_info_lut["pre_flop"] = compute_preflop_lossy_abstraction(
                builder=self
            )
            with open(self.card_info_lut_path, 'wb') as f:
                pickle.dump(self.card_info_lut, f)
        if "river" not in self.card_info_lut:
            self.card_info_lut["river"] = self._compute_river_clusters(
                n_river_clusters,
            )
            with open(self.card_info_lut_path, 'wb') as f:
                pickle.dump(self.card_info_lut, f)
            with open(self.centroid_path, 'wb') as f:
                pickle.dump(self.centroids, f)
        if "turn" not in self.card_info_lut:
            self.load_turn()
            self.card_info_lut["turn"] = self._compute_turn_clusters(n_turn_clusters)
            with open(self.card_info_lut_path, 'wb') as f:
                pickle.dump(self.card_info_lut, f)
            with open(self.centroid_path, 'wb') as f:
                pickle.dump(self.centroids, f)
        if "flop" not in self.card_info_lut:
            self.load_flop()
            self.card_info_lut["flop"] = self._compute_flop_clusters(n_flop_clusters)
            with open(self.card_info_lut_path, 'wb') as f:
                pickle.dump(self.card_info_lut, f)
            with open(self.centroid_path, 'wb') as f:
                pickle.dump(self.centroids, f)
        end = time.time()
        log.info(f"Finished computation of clusters - took {end - start} seconds.")

    def _compute_river_clusters(self, n_river_clusters: int):
        """Compute river clusters and create lookup table."""
        log.info("Starting computation of river clusters.")
        start = time.time()
        self.load_river()
        river_ehs_sm = None
        river_size = math.comb(len(self._cards), 2) * math.comb(len(self._cards) - 2, 5)
        try:
            with open(self.ehs_river_path, 'rb') as f:
                river_ehs = pickle.load(f)
            log.info("loaded river ehs")
        except FileNotFoundError:
            river_ehs = multiprocess_ehs_calc(
                self.river, lambda public: process_river_ehs(public, self._evaluator, self.n_simulations_river), river_size
            )
            with open(self.ehs_river_path, 'wb') as f:
                pickle.dump(river_ehs, f)

        self.centroids["river"], self._river_clusters = self.cluster(
            num_clusters=n_river_clusters, X=river_ehs
        )
        end = time.time()
        log.info(
            f"Finished computation of river clusters - took {end - start} seconds."
        )
        if river_ehs_sm is not None:
            river_ehs_sm.close()
            river_ehs_sm.unlink()
        self.load_river()
        return self.create_card_lookup(self._river_clusters, self.river, river_size)

    def _compute_turn_clusters(self, n_turn_clusters: int):
        """Compute turn clusters and create lookup table."""
        log.info("Starting computation of turn clusters.")
        start = time.time()
        ehs_sm = None

        def batch_tasker(batch, cursor, result):
            for i, x in enumerate(batch):
                result[cursor + i] = self.process_turn_ehs_distributions(x)

        self._turn_ehs_distributions, ehs_sm = multiprocess_ehs_calc(
            iter(self.turn),
            batch_tasker,
            len(self.turn),
            len(self.centroids["river"]),
        )

        self.centroids["turn"], self._turn_clusters = self.cluster(
            num_clusters=n_turn_clusters, X=self._turn_ehs_distributions
        )
        end = time.time()
        log.info(f"Finished computation of turn clusters - took {end - start} seconds.")

        ehs_sm.close()
        ehs_sm.unlink()

        return self.create_card_lookup(self._turn_clusters, self.turn)

    def _compute_flop_clusters(self, n_flop_clusters: int):
        """Compute flop clusters and create lookup table."""
        log.info("Starting computation of flop clusters.")
        start = time.time()
        ehs_sm = None

        def batch_tasker(batch, cursor, result):
            for i, x in enumerate(batch):
                result[cursor + i] = (
                    self.process_flop_potential_aware_distributions(x)
                )

        self._flop_potential_aware_distributions, ehs_sm = multiprocess_ehs_calc(
            iter(self.flop),
            batch_tasker,
            len(self.flop),
            len(self.centroids["turn"]),
        )

        self.centroids["flop"], self._flop_clusters = self.cluster(
            num_clusters=n_flop_clusters, X=self._flop_potential_aware_distributions
        )
        end = time.time()
        log.info(f"Finished computation of flop clusters - took {end - start} seconds.")

        ehs_sm.close()
        ehs_sm.unlink()

        return self.create_card_lookup(self._flop_clusters, self.flop)

    def simulate_get_ehs(self, game: GameUtility,) -> np.ndarray:
        """
        Get expected hand strength object.

        Parameters
        ----------
        game : GameUtility
            GameState for help with determining winner and sampling opponent hand

        Returns
        -------
        ehs : np.ndarray
            [win_rate, loss_rate, tie_rate]
        """
        ehs: np.ndarray = np.zeros(3)
        for _ in range(self.n_simulations_river):
            idx: int = game.get_winner()
            # increment win rate for winner/tie
            ehs[idx] += 1 / self.n_simulations_river
        return ehs

    def simulate_get_turn_ehs_distributions(
        self,
        public: np.ndarray,
    ) -> np.ndarray:
        available_cards = np.array([c for c in self._cards if c not in public])
        hand_size = len(public) + 1
        hand_evaluator = self._evaluator.hand_size_map[hand_size]

        prob_unit = 1 / self.n_simulations_turn
        prob_sub_unit = 1 / self.n_simulations_river
        ehs: np.ndarray = np.zeros(3)
        our_hand = np.zeros(len(public) + 1, dtype=int)
        our_hand[:len(public)] = public
        opp_hand = np.zeros(len(public) + 1, dtype=int)
        opp_hand[:len(public)] = public

        turn_ehs_distribution = np.zeros(len(self.centroids["river"]))
        for _ in range(self.n_simulations_turn):
            river_card = np.random.choice(available_cards, 1, replace=False)
            non_river_cards = np.array([c for c in available_cards if c != river_card])
            our_hand[-1:] = river_card
            opp_hand[-1:] = river_card

            for _ in range(self.n_simulations_river):
                opp_hand[:2] = np.random.choice(non_river_cards, 2, replace=False)
                our_hand_rank = hand_evaluator(our_hand)
                opp_hand_rank = hand_evaluator(opp_hand)
                if our_hand_rank > opp_hand_rank:
                    ehs[0] += prob_sub_unit
                elif our_hand_rank < opp_hand_rank:
                    ehs[1] += prob_sub_unit
                else:
                    ehs[2] += prob_sub_unit

            for idx, river_centroid in enumerate(self.centroids["river"]):
                emd = wasserstein_distance(ehs, river_centroid)
                if idx == 0 or emd < min_emd:
                    min_idx = idx
                    min_emd = emd
            turn_ehs_distribution[min_idx] += prob_unit

        return turn_ehs_distribution


    @staticmethod
    def get_available_cards(
        cards: np.ndarray, unavailable_cards: np.ndarray
    ) -> np.ndarray:
        unavailable_cards = set(unavailable_cards)
        return np.array([c for c in cards if c not in unavailable_cards])

    def process_turn_ehs_distributions(self, public: np.ndarray) -> np.ndarray:
        turn_ehs_distribution = self.simulate_get_turn_ehs_distributions(public)
        return turn_ehs_distribution

    def process_flop_potential_aware_distributions(
        self, public: np.ndarray,
    ) -> np.ndarray:
        available_cards: np.ndarray = self.get_available_cards(
            cards=self._cards, unavailable_cards=public
        )
        potential_aware_distribution_flop = np.zeros(len(self.centroids["turn"]))
        extended_public = np.zeros(len(public) + 1)
        extended_public[:-1] = public
        for j in range(self.n_simulations_flop):
            turn_card = np.random.choice(available_cards, 1, replace=False)
            extended_public[-1:] = turn_card
            available_cards_turn = np.array(
                [x for x in available_cards if x != turn_card[0]]
            )
            turn_ehs_distribution = self.simulate_get_turn_ehs_distributions(
                extended_public,
            )
            for idx, turn_centroid in enumerate(self.centroids["turn"]):
                emd = wasserstein_distance(turn_ehs_distribution, turn_centroid)
                if idx == 0 or emd < min_emd:
                    min_idx = idx
                    min_emd = emd
            potential_aware_distribution_flop[min_idx] += 1 / self.n_simulations_flop
        return potential_aware_distribution_flop

    @staticmethod
    def cluster(num_clusters: int, X: np.ndarray):
        km = MiniBatchKMeans(
            n_clusters=num_clusters,
            batch_size=1000,
            init="random",
            n_init=10,
            max_iter=300,
            tol=1e-04,
            random_state=0,
        )
        y_km = km.fit_predict(X)
        centroids = km.cluster_centers_
        return centroids, y_km

    @staticmethod
    def create_card_lookup(
        clusters: np.ndarray,
        card_combos: np.ndarray,
        card_combos_size: Optional[int] = None,
    ) -> Dict:
        log.info("Creating lookup table.")
        lossy_lookup = {}
        for i, card_combo in enumerate(tqdm(card_combos, ascii=" >=", total=card_combos_size)):
            lossy_lookup[tuple(card_combo)] = clusters[i]
        return lossy_lookup
