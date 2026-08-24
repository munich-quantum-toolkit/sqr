# Copyright (c) 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for initial placement strategies."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import networkx as nx
import pytest

from mqt.sqr.placements.interaction_placement_strategy import DECAY, InteractionPlacementStrategy
from mqt.sqr.placements.random_strategy import RandomPlacementStrategy
from mqt.sqr.placements.reverse_traversal_strategy import ReverseTraversalPlacementStrategy
from mqt.sqr.routing.rotation_routing import RotationRoutingPlanner
from mqt.sqr.utils.network import NetworkBuilder

if TYPE_CHECKING:
    from mqt.sqr.routing.common import Qubit


def make_sn_network(sn_nodes: list[tuple[int, int]]) -> nx.Graph:
    """Create a minimal network containing only stabilizer nodes.

    Returns:
        The created network.
    """
    network = nx.Graph()
    network.add_nodes_from((node, {"type": "SN"}) for node in sn_nodes)
    return network


def pair_ids(pairs: list[tuple[Qubit, Qubit]]) -> list[tuple[int, int]]:
    """Return the qubit IDs of a list of qubit pairs."""
    return [(first.id, second.id) for first, second in pairs]


def test_build_pairs_is_deterministic() -> None:
    """The same seed should produce the same interaction pairs."""
    strategy = RandomPlacementStrategy()

    first = strategy.build_pairs(
        n_qubits=6,
        rounds=3,
        seed=42,
    )
    second = strategy.build_pairs(
        n_qubits=6,
        rounds=3,
        seed=42,
    )

    assert first == second


def test_build_pairs_respects_max_pairs_per_round() -> None:
    """No round should contain more pairs than requested."""
    strategy = RandomPlacementStrategy()
    rounds = 4
    max_pairs_per_round = 2

    pairs = strategy.build_pairs(
        n_qubits=8,
        rounds=rounds,
        max_pairs_per_round=max_pairs_per_round,
        seed=42,
    )

    assert len(pairs) == rounds * max_pairs_per_round


def test_build_pairs_uses_each_qubit_at_most_once_per_round() -> None:
    """A qubit must not occur in two pairs of the same round."""
    strategy = RandomPlacementStrategy()
    max_pairs_per_round = 2

    pairs = strategy.build_pairs(
        n_qubits=6,
        rounds=3,
        max_pairs_per_round=max_pairs_per_round,
        seed=42,
    )

    for start in range(0, len(pairs), max_pairs_per_round):
        round_pairs = pairs[start : start + max_pairs_per_round]
        qubit_ids = [qubit_id for pair in round_pairs for qubit_id in pair]

        assert len(qubit_ids) == len(set(qubit_ids))


def test_random_placement_is_deterministic() -> None:
    """Random placement should be reproducible for a fixed seed."""
    strategy = RandomPlacementStrategy()
    sn_nodes = [(0, 0), (0, 1), (1, 0), (1, 1)]

    first = strategy.place_qubits(
        sn_nodes=sn_nodes,
        n_qubits=3,
        seed=42,
    )
    second = strategy.place_qubits(
        sn_nodes=sn_nodes,
        n_qubits=3,
        seed=42,
    )

    assert first == second
    assert len(first) == 3
    assert len(set(first)) == 3
    assert set(first) <= set(sn_nodes)


def test_interaction_placement_falls_back_to_random_placement() -> None:
    """Without interaction data, interaction placement should behave randomly."""
    interaction_strategy = InteractionPlacementStrategy()
    random_strategy = RandomPlacementStrategy()
    sn_nodes = [(0, 0), (0, 1), (1, 0), (1, 1)]

    interaction_placement = interaction_strategy.place_qubits(
        sn_nodes=sn_nodes,
        n_qubits=3,
        seed=42,
    )
    random_placement = random_strategy.place_qubits(
        sn_nodes=sn_nodes,
        n_qubits=3,
        seed=42,
    )

    assert interaction_placement == random_placement


def test_interaction_placement_applies_decay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Repeated interactions should be accumulated using the decay factor."""
    strategy = InteractionPlacementStrategy()

    # With two qubits, every round necessarily produces the same interaction.
    strategy.build_pairs(
        n_qubits=2,
        rounds=3,
        seed=42,
    )

    captured_graph: nx.Graph | None = None

    def fake_spring_layout(
        graph: nx.Graph,
        weight: str,
        seed: int | None,
    ) -> dict[int, tuple[float, float]]:
        nonlocal captured_graph
        captured_graph = graph.copy()

        assert weight == "weight"
        assert seed == 42

        return {
            0: (0.0, 0.0),
            1: (1.0, 0.0),
        }

    monkeypatch.setattr(nx, "spring_layout", fake_spring_layout)

    strategy.place_qubits(
        sn_nodes=[(0, 0), (1, 0)],
        n_qubits=2,
        seed=42,
    )

    assert captured_graph is not None

    expected_weight = 1.0 + DECAY + DECAY**2
    assert captured_graph[0][1]["weight"] == pytest.approx(expected_weight)


def test_interaction_placement_maps_qubits_according_to_layout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Qubits should be assigned to SN nodes according to the layout order."""
    strategy = InteractionPlacementStrategy()
    strategy.build_pairs(
        n_qubits=3,
        rounds=1,
        seed=42,
    )

    positions = {
        0: (2.0, 0.0),
        1: (0.0, 0.0),
        2: (1.0, 0.0),
    }

    monkeypatch.setattr(
        nx,
        "spring_layout",
        lambda *_args, **_kwargs: positions,
    )

    placement = strategy.place_qubits(
        sn_nodes=[(1, 0), (0, 1), (0, 0)],
        n_qubits=3,
        seed=42,
    )

    # Layout order: qubit 1, qubit 2, qubit 0
    # SN-node order: (0, 0), (0, 1), (1, 0)
    assert placement == [
        (1, 0),  # qubit 0
        (0, 0),  # qubit 1
        (0, 1),  # qubit 2
    ]


def test_build_network_and_place_rejects_too_many_qubits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Placement should fail if the network has too few stabilizer nodes."""
    network = make_sn_network([(0, 0), (1, 0)])

    monkeypatch.setattr(
        NetworkBuilder,
        "build_network",
        staticmethod(lambda _width, _height: network),
    )

    strategy = RandomPlacementStrategy()

    with pytest.raises(
        ValueError,
        match=r"n_qubits=3 exceeds available SN nodes \(2\)",
    ):
        strategy.build_network_and_place(
            width=1,
            height=1,
            n_qubits=3,
            rounds=1,
            seed=42,
        )


def test_reverse_traversal_routes_pairs_in_reverse_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The warm-up routing should process interactions in reverse order."""
    sn_nodes = [(0, 0), (0, 1), (1, 0), (1, 1)]
    network = make_sn_network(sn_nodes)

    monkeypatch.setattr(
        NetworkBuilder,
        "build_network",
        staticmethod(lambda _width, _height: network),
    )

    strategy = ReverseTraversalPlacementStrategy()

    expected_pairs = strategy.build_pairs(
        n_qubits=4,
        rounds=2,
        seed=42,
    )

    routed_pair_ids: list[tuple[int, int]] = []

    def fake_route(
        self: RotationRoutingPlanner,
        network: nx.Graph,
        qubits: list[Qubit],
        pairs: list[tuple[Qubit, Qubit]],
        p_success: float,
        p_repair: float,
    ) -> tuple[dict[int, list[tuple[tuple[int, int], Any]]], None]:
        del self, network, qubits

        routed_pair_ids.extend(pair_ids(pairs))

        assert p_success == pytest.approx(1.0)
        assert p_repair == pytest.approx(1.0)

        timelines = {qubit_id: [((qubit_id, 10), None)] for qubit_id in range(4)}
        return timelines, None

    monkeypatch.setattr(RotationRoutingPlanner, "route", fake_route)

    strategy.build_network_and_place(
        width=2,
        height=2,
        n_qubits=4,
        rounds=2,
        seed=42,
    )

    assert routed_pair_ids == list(reversed(expected_pairs))


def test_reverse_traversal_preserves_original_pair_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Returned interaction pairs should retain their original order."""
    sn_nodes = [(0, 0), (0, 1), (1, 0), (1, 1)]
    network = make_sn_network(sn_nodes)

    monkeypatch.setattr(
        NetworkBuilder,
        "build_network",
        staticmethod(lambda _width, _height: network),
    )

    def fake_route(
        self: RotationRoutingPlanner,
        network: nx.Graph,
        qubits: list[Qubit],
        pairs: list[tuple[Qubit, Qubit]],
        p_success: float,
        p_repair: float,
    ) -> tuple[dict[int, list[tuple[tuple[int, int], Any]]], None]:
        del self, network, qubits, pairs, p_success, p_repair

        timelines = {qubit_id: [((qubit_id, 10), None)] for qubit_id in range(4)}
        return timelines, None

    monkeypatch.setattr(RotationRoutingPlanner, "route", fake_route)

    strategy = ReverseTraversalPlacementStrategy()

    expected_pairs = strategy.build_pairs(
        n_qubits=4,
        rounds=2,
        seed=42,
    )

    _, qubits, pairs = strategy.build_network_and_place(
        width=2,
        height=2,
        n_qubits=4,
        rounds=2,
        seed=42,
    )

    assert [qubit.id for qubit in qubits] == [0, 1, 2, 3]
    assert pair_ids(pairs) == expected_pairs
