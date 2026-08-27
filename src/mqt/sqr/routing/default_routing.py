# Copyright (c) 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

from __future__ import annotations

import itertools
import random
from collections import deque
from copy import deepcopy

import networkx as nx

from mqt.sqr.routing.common import MAX_TIME, AStar, Coord, Qubit, Reservations, TimedNode
from mqt.sqr.routing.routing_strategy import RoutingResult, RoutingStrategy

MAX_REPLANS = 50
MAX_GLOBAL_ITERS = 50


class DefaultRoutingPlanner(RoutingStrategy):
    def route(
        self,
        graph: nx.Graph,
        qubits: list[Qubit],
        pairs: list[tuple[Qubit, Qubit]],
        p_success: float,
        p_repair: float,
    ) -> RoutingResult:
        current_pos: dict[int, Coord] = {q.id: q.pos for q in qubits}
        all_qids: set[int] = {q.id for q in qubits}

        defective_edges: set[frozenset] = set()
        batch_plans: list[dict[int, list[TimedNode]]] = []
        batch_defects: list[set[frozenset]] = []

        total_ins: set[Coord] = {n for n in graph if graph.nodes[n].get("type") == "IN"}
        tried_meetings: dict[frozenset, set[Coord]] = {}

        layers = self._build_layers(pairs)

        idx = 0
        replan_counts: dict[int, int] = {}
        global_iter = 0

        while idx < len(layers):
            global_iter += 1
            if global_iter > MAX_GLOBAL_ITERS:
                msg = (
                    f"Aborted (safeguard): more than {MAX_GLOBAL_ITERS} iterations "
                    f"in the main routing loop (possible infinite loop, current layer = {idx})."
                )
                raise RuntimeError(msg)

            tried_meetings.clear()

            layer_pairs = layers[idx]
            layer_qids: set[int] = {x for ab in layer_pairs for x in ab}
            non_layer_qids: set[int] = all_qids - layer_qids
            layer_starts: set[Coord] = {current_pos[q] for q in layer_qids}
            occupied_now: set[Coord] = {current_pos[q] for q in all_qids}

            replan_current_layer = False

            (
                to_meeting_plans,
                fixed_meetings,
                _,
                unplaceable_pairs_step1,
                exhausted_pairs_step1,
            ) = DefaultRoutingPlanner.plan_layer_only(
                graph=graph,
                current_pos=current_pos,
                layer_pairs=layer_pairs,
                layer_starts=layer_starts,
                defective_edges=defective_edges,
                banned_meetings=tried_meetings,
                all_ins=total_ins,
            )

            if exhausted_pairs_step1:
                layers[idx + 1 : idx + 1] = [exhausted_pairs_step1]

            if unplaceable_pairs_step1:
                layers[idx + 1 : idx + 1] = [unplaceable_pairs_step1]
                if not fixed_meetings:
                    wait = {qid: [(current_pos[qid], 0), (current_pos[qid], 1)] for qid in all_qids}
                    batch_plans.append(wait)
                    DefaultRoutingPlanner._snapshot_defects(batch_defects, defective_edges, 1)
                    idx += 1
                    continue

            if not fixed_meetings and not unplaceable_pairs_step1 and not exhausted_pairs_step1:
                msg = f"Layer {idx} unsolvable: no meeting-INs fixed and no spillover possible. Pairs: {layer_pairs}"
                raise RuntimeError(msg)

            if not fixed_meetings:
                wait = {qid: [(current_pos[qid], 0), (current_pos[qid], 1)] for qid in all_qids}
                batch_plans.append(wait)
                DefaultRoutingPlanner._snapshot_defects(batch_defects, defective_edges, 1)
                idx += 1
                continue

            f_layer: set[Coord] = DefaultRoutingPlanner.collect_layer_nodes(to_meeting_plans, fixed_meetings)
            f_all = set(f_layer) | set(layer_starts)

            blockers_now: list[int] = [qid for qid in non_layer_qids if current_pos[qid] in f_all]

            blocker_to_pair: dict[int, tuple[int, int]] = {}
            evac_plans: dict[int, list[TimedNode]] = {}

            if blockers_now:
                node_to_pairs: dict[Coord, list[tuple[int, int]]] = {}
                for a, b in layer_pairs:
                    key = frozenset({a, b})
                    if key not in fixed_meetings:
                        continue
                    nodes = DefaultRoutingPlanner.path_nodes_of_pair(to_meeting_plans, a, b)
                    for n in nodes:
                        node_to_pairs.setdefault(n, []).append((a, b))

                for qid in blockers_now:
                    pos = current_pos[qid]
                    pairs_touching = node_to_pairs.get(pos, [])
                    if pairs_touching:
                        blocker_to_pair[qid] = pairs_touching[0]

                avoid_for_targets = set(occupied_now) | f_all
                targets: dict[int, Coord] = {}
                for qid in blockers_now:
                    tgt = DefaultRoutingPlanner.nearest_free_sn(graph, current_pos[qid], avoid_for_targets)
                    if tgt is not None and tgt not in f_layer:
                        targets[qid] = tgt
                        avoid_for_targets.add(tgt)

                cannot_place = [qid for qid in blockers_now if qid not in targets]
                newly_affected: list[tuple[int, int]] = []
                for qid in cannot_place:
                    ab = blocker_to_pair.get(qid)
                    if ab:
                        newly_affected.append(ab)

                if newly_affected:
                    unique_pairs: list[tuple[int, int]] = []
                    seen_pairs: set[frozenset] = set()
                    for ab in newly_affected:
                        pkey = frozenset(ab)
                        if pkey in seen_pairs:
                            continue
                        seen_pairs.add(pkey)
                        unique_pairs.append(ab)
                        to_meeting_plans.pop(ab[0], None)
                        to_meeting_plans.pop(ab[1], None)
                        fixed_meetings.pop(pkey, None)
                    replan_current_layer = True

                evacuating = {qid: current_pos[qid] for qid in blockers_now if qid in targets}
                if evacuating:
                    try:
                        evac_plans = DefaultRoutingPlanner.mapf_to_targets(
                            graph=graph,
                            starts=evacuating,
                            targets={qid: targets[qid] for qid in evacuating},
                            blocked_nodes=f_all,
                            blocked_edges=defective_edges,
                        )
                    except RuntimeError:
                        evac_plans = {}
                        blocked_now = set(f_all)
                        for qid in evacuating:
                            try:
                                one = DefaultRoutingPlanner.mapf_to_targets(
                                    graph=graph,
                                    starts={qid: current_pos[qid]},
                                    targets={qid: targets[qid]},
                                    blocked_nodes=blocked_now,
                                    blocked_edges=defective_edges,
                                )
                                evac_plans[qid] = one[qid]
                                blocked_now.add(one[qid][-1][0])
                            except RuntimeError:
                                ab = blocker_to_pair.get(qid)
                                if ab:
                                    pkey = frozenset(ab)
                                    to_meeting_plans.pop(ab[0], None)
                                    to_meeting_plans.pop(ab[1], None)
                                    fixed_meetings.pop(pkey, None)
                                    replan_current_layer = True

                if evac_plans:
                    waiting_qids = non_layer_qids - set(evacuating.keys())
                    evac_plans = DefaultRoutingPlanner._resolve_evacuate_collisions_with_waiters(
                        graph=graph,
                        evac_plans=evac_plans,
                        targets={qid: targets[qid] for qid in evacuating},
                        current_pos=current_pos,
                        waiting_qids=waiting_qids,
                        blocked_nodes=f_all,
                        defective_edges=defective_edges,
                        blocker_to_pair=blocker_to_pair,
                    )

            if replan_current_layer:
                replan_counts[idx] = replan_counts.get(idx, 0) + 1
                if replan_counts[idx] > MAX_REPLANS:
                    msg = f"No valid routing for layer {idx} after {replan_counts[idx]} replans."
                    raise RuntimeError(msg)
                continue

            if not fixed_meetings:
                wait = {qid: [(current_pos[qid], 0), (current_pos[qid], 1)] for qid in all_qids}
                batch_plans.append(wait)
                DefaultRoutingPlanner._snapshot_defects(batch_defects, defective_edges, 1)
                idx += 1
                continue

            DefaultRoutingPlanner.sample_edge_failures(
                graph, defective_edges, p_fail=(1.0 - p_success), p_repair=p_repair
            )

            if evac_plans:
                to_spill_for_nonlayer: set[tuple[int, int]] = set()
                for qid, path in evac_plans.items():
                    if DefaultRoutingPlanner.path_uses_defective_edge(path, defective_edges):
                        ab = blocker_to_pair.get(qid)
                        if ab:
                            to_spill_for_nonlayer.add(ab)

                if to_spill_for_nonlayer:
                    for a, b in to_spill_for_nonlayer:
                        key = frozenset({a, b})
                        to_meeting_plans.pop(a, None)
                        to_meeting_plans.pop(b, None)
                        fixed_meetings.pop(key, None)
                    layers[idx + 1 : idx + 1] = [list(to_spill_for_nonlayer)]
                    evac_plans.clear()

            if not fixed_meetings:
                wait = {qid: [(current_pos[qid], 0), (current_pos[qid], 1)] for qid in all_qids}
                batch_plans.append(wait)
                DefaultRoutingPlanner._snapshot_defects(batch_defects, defective_edges, 1)
                idx += 1
                continue

            pre_in_paths: dict[int, list[TimedNode]] = {}
            t_pre_sync = 0

            for a, b in layer_pairs:
                key = frozenset({a, b})
                if key not in fixed_meetings:
                    continue
                meet = fixed_meetings[key]

                for qid in (a, b):
                    cut = DefaultRoutingPlanner.retime_until_pre_in_wait(to_meeting_plans[qid], meet, 0)
                    if cut is None:
                        to_meeting_plans.pop(a, None)
                        to_meeting_plans.pop(b, None)
                        fixed_meetings.pop(key, None)
                        replan_current_layer = True
                        break

                    pre_in_paths[qid] = cut
                    if cut:
                        t_pre_sync = max(t_pre_sync, cut[-1][1])

            if replan_current_layer:
                continue

            if not fixed_meetings:
                wait = {qid: [(current_pos[qid], 0), (current_pos[qid], 1)] for qid in all_qids}
                batch_plans.append(wait)
                DefaultRoutingPlanner._snapshot_defects(batch_defects, defective_edges, 1)
                idx += 1
                continue

            to_spill_layer_defects: list[tuple[int, int]] = []
            for a, b in layer_pairs:
                key = frozenset({a, b})
                if key not in fixed_meetings:
                    continue

                meet = fixed_meetings[key]
                pa = pre_in_paths.get(a)
                pb = pre_in_paths.get(b)

                if pa is None or pb is None:
                    to_spill_layer_defects.append((a, b))
                    continue

                if DefaultRoutingPlanner.path_uses_defective_edge(pa, defective_edges):
                    to_spill_layer_defects.append((a, b))
                    continue

                if DefaultRoutingPlanner.path_uses_defective_edge(pb, defective_edges):
                    to_spill_layer_defects.append((a, b))
                    continue

                pre_a = pa[-1][0]
                pre_b = pb[-1][0]
                if frozenset({pre_a, meet}) in defective_edges or frozenset({pre_b, meet}) in defective_edges:
                    to_spill_layer_defects.append((a, b))

            if to_spill_layer_defects:
                for a, b in to_spill_layer_defects:
                    key = frozenset({a, b})
                    to_meeting_plans.pop(a, None)
                    to_meeting_plans.pop(b, None)
                    fixed_meetings.pop(key, None)
                layers[idx + 1 : idx + 1] = [to_spill_layer_defects]

            if not fixed_meetings:
                wait = {qid: [(current_pos[qid], 0), (current_pos[qid], 1)] for qid in all_qids}
                batch_plans.append(wait)
                DefaultRoutingPlanner._snapshot_defects(batch_defects, defective_edges, 1)
                idx += 1
                continue

            if evac_plans:
                evac_plans = {
                    qid: path
                    for qid, path in evac_plans.items()
                    if not DefaultRoutingPlanner.path_uses_defective_edge(path, defective_edges)
                }

                micro_evacuate: dict[int, list[TimedNode]] = dict(evac_plans.items())
                dur = max((p[-1][1] for p in micro_evacuate.values()), default=0)

                for qid in all_qids - set(micro_evacuate.keys()):
                    micro_evacuate[qid] = [(current_pos[qid], 0), (current_pos[qid], dur)]

                batch_plans.append(micro_evacuate)
                DefaultRoutingPlanner._snapshot_defects(batch_defects, defective_edges, 1)

                for qid in evac_plans:
                    current_pos[qid] = evac_plans[qid][-1][0]

            micro_to_pre: dict[int, list[TimedNode]] = {}
            exec_layer_qids: set[int] = set()

            for a, b in layer_pairs:
                key = frozenset({a, b})
                if key not in fixed_meetings:
                    continue
                meet = fixed_meetings[key]

                for qid in (a, b):
                    cut = DefaultRoutingPlanner.retime_until_pre_in_wait(to_meeting_plans[qid], meet, t_pre_sync)
                    assert cut is not None
                    micro_to_pre[qid] = cut
                    exec_layer_qids.add(qid)

            others = all_qids - exec_layer_qids
            dur_pre = max((p[-1][1] for p in micro_to_pre.values()), default=0)
            for qid in others:
                micro_to_pre[qid] = [(current_pos[qid], 0), (current_pos[qid], dur_pre)]

            batch_plans.append(micro_to_pre)
            DefaultRoutingPlanner._snapshot_defects(batch_defects, defective_edges, 1)

            for qid in exec_layer_qids:
                current_pos[qid] = micro_to_pre[qid][-1][0]

            micro_in: dict[int, list[TimedNode]] = {}
            active_layer_qids = {qid for pair in layer_pairs if frozenset(pair) in fixed_meetings for qid in pair}

            for a, b in layer_pairs:
                key = frozenset({a, b})
                if key not in fixed_meetings:
                    continue

                meet = fixed_meetings[key]
                pre_a = current_pos[a]
                pre_b = current_pos[b]

                if frozenset({pre_a, meet}) in defective_edges or frozenset({pre_b, meet}) in defective_edges:
                    micro_in[a] = [(pre_a, 0), (pre_a, 2)]
                    micro_in[b] = [(pre_b, 0), (pre_b, 2)]
                else:
                    micro_in[a] = [(pre_a, 0), (meet, 1), (pre_a, 2)]
                    micro_in[b] = [(pre_b, 0), (meet, 1), (pre_b, 2)]

            for qid in all_qids - active_layer_qids:
                micro_in[qid] = [(current_pos[qid], 0), (current_pos[qid], 2)]

            batch_plans.append(micro_in)
            DefaultRoutingPlanner._snapshot_defects(batch_defects, defective_edges, 1)

            idx += 1

        return DefaultRoutingPlanner.stitch_batches(qubits, batch_plans, batch_defects)

    @staticmethod
    def _build_layers(pairs: list[tuple[Qubit, Qubit]]) -> list[list[tuple[int, int]]]:
        layers: list[list[tuple[int, int]]] = []
        used: set[int] = set()
        cur: list[tuple[int, int]] = []

        for qa, qb in pairs:
            a, b = qa.id, qb.id
            if a not in used and b not in used:
                cur.append((a, b))
                used |= {a, b}
            else:
                if cur:
                    layers.append(cur)
                cur = [(a, b)]
                used = {a, b}

        if cur:
            layers.append(cur)

        return layers

    @staticmethod
    def _snapshot_defects(
        batch_defects: list[set[frozenset]],
        defective_edges: set[frozenset],
        n: int,
    ) -> None:
        batch_defects.extend(set(defective_edges) for _ in range(n))

    @staticmethod
    def _forbidden_for_layer_qid(
        current_pos: dict[int, Coord],
        layer_starts: set[Coord],
        qid: int,
    ) -> set[Coord]:
        return set(layer_starts) - {current_pos[qid]}

    @staticmethod
    def _chebyshev(a: Coord, b: Coord) -> int:
        return max(abs(a[0] - b[0]), abs(a[1] - b[1]))

    @staticmethod
    def _build_blocker_chain_along_path(
        path: list[TimedNode],
        wait_pos: dict[int, Coord],
        blocker_to_pair: dict[int, tuple[int, int]],
        root_pair: tuple[int, int] | None,
    ) -> list[int]:
        chain: list[int] = []
        seen: set[int] = set()

        for (_, _), (node_next, _) in itertools.pairwise(path):
            for bqid, bpos in wait_pos.items():
                if bqid in seen:
                    continue
                if node_next == bpos:
                    chain.append(bqid)
                    seen.add(bqid)
                    if root_pair is not None:
                        blocker_to_pair.setdefault(bqid, root_pair)

        return chain

    @staticmethod
    def _block_non_sn_nodes(
        res: Reservations,
        graph: nx.Graph,
        allowed: set[Coord],
    ) -> None:
        for node in graph.nodes:
            if node in allowed:
                continue
            if graph.nodes[node].get("type") != "SN":
                cap = res.node_capacity(node)
                for t in range(MAX_TIME + 1):
                    res.node_caps[node][t] = cap

    @staticmethod
    def plan_layer_only(
        graph: nx.Graph,
        current_pos: dict[int, Coord],
        layer_pairs: list[tuple[int, int]],
        layer_starts: set[Coord],
        defective_edges: set[frozenset],
        banned_meetings: dict[frozenset, set[Coord]] | None = None,
        all_ins: set[Coord] | None = None,
    ) -> tuple[
        dict[int, list[TimedNode]],
        dict[frozenset, Coord],
        bool,
        list[tuple[int, int]],
        list[tuple[int, int]],
    ]:
        res = Reservations(graph, blocked_edges=defective_edges)
        plans: dict[int, list[TimedNode]] = {}
        fixed_meetings: dict[frozenset, Coord] = {}
        unplaceable: list[tuple[int, int]] = []

        banned_meetings = banned_meetings or {}
        all_ins = all_ins or set()
        exhausted_pairs: list[tuple[int, int]] = []

        cand_per_pair: dict[tuple[int, int], list[Coord]] = {}
        for a, b in layer_pairs:
            qa, qb = current_pos[a], current_pos[b]
            cand_per_pair[a, b] = DefaultRoutingPlanner.best_meeting_candidates(
                graph, qa, qb, reserved=set(), forbidden_nodes=set()
            )

        order = sorted(layer_pairs, key=lambda ab: len(cand_per_pair.get(ab, [])))
        reserved_in: set[Coord] = set()

        for a, b in order:
            qa, qb = current_pos[a], current_pos[b]
            placed = False

            cur_preins_map = DefaultRoutingPlanner._preins_for_plans(plans, fixed_meetings)
            existing_preins: set[Coord] = set(cur_preins_map.values()) if cur_preins_map else set()

            cands_all = DefaultRoutingPlanner.best_meeting_candidates(
                graph, qa, qb, reserved=reserved_in, forbidden_nodes=set()
            )
            banned = banned_meetings.get(frozenset({a, b}), set())
            cands = [c for c in cands_all if c not in banned]

            existing_path_nodes: set[Coord] = set()
            for p in plans.values():
                existing_path_nodes.update(c for c, _ in p)

            if not cands and all_ins and len(banned) >= len(all_ins):
                exhausted_pairs.append((a, b))

            for meet in cands:
                res_try = deepcopy(res)

                allowed_a = {qa, meet}
                allowed_b = {qb, meet}
                DefaultRoutingPlanner._block_non_sn_nodes(res_try, graph, allowed=allowed_a | allowed_b)

                if existing_preins:
                    DefaultRoutingPlanner._block_nodes(res_try, existing_preins)

                DefaultRoutingPlanner._block_nodes(
                    res_try,
                    DefaultRoutingPlanner._forbidden_for_layer_qid(current_pos, layer_starts, a),
                )
                pa = AStar.search(graph, qa, meet, res_try)
                if pa is None:
                    continue
                Reservations.commit(res_try, pa)

                pre_a = DefaultRoutingPlanner.entry_sn_from_path(pa, meet)
                if pre_a is None:
                    continue
                if pre_a in existing_path_nodes:
                    continue

                DefaultRoutingPlanner._block_nodes(res_try, {pre_a})

                if existing_preins:
                    DefaultRoutingPlanner._block_nodes(res_try, existing_preins)

                DefaultRoutingPlanner._block_nodes(
                    res_try,
                    DefaultRoutingPlanner._forbidden_for_layer_qid(current_pos, layer_starts, b),
                )
                pb = AStar.search(graph, qb, meet, res_try)
                if pb is None:
                    continue

                pre_b = DefaultRoutingPlanner.entry_sn_from_path(pb, meet)
                if pre_b is None:
                    continue
                if pre_b in existing_path_nodes:
                    continue

                tmp_plans = dict(plans)
                tmp_plans[a] = pa
                tmp_plans[b] = pb

                tmp_fixed = dict(fixed_meetings)
                tmp_fixed[frozenset({a, b})] = meet

                preins = DefaultRoutingPlanner._preins_for_plans(tmp_plans, tmp_fixed)
                if preins is None:
                    continue

                layer_ids = {x for ab in layer_pairs for x in ab}
                prein_vals = [preins[qid] for qid in preins if qid in layer_ids]
                if len(set(prein_vals)) != len(prein_vals):
                    continue

                Reservations.commit(res_try, pb)

                res = res_try
                plans[a] = pa
                plans[b] = pb
                fixed_meetings[frozenset({a, b})] = meet
                reserved_in.add(meet)
                placed = True
                break

            if not placed:
                unplaceable.append((a, b))

        preins_ok = True
        preins = DefaultRoutingPlanner._preins_for_plans(plans, fixed_meetings)
        if preins is None:
            preins_ok = False
        else:
            layer_ids = {x for ab in layer_pairs for x in ab}
            lv = [preins[qid] for qid in preins if qid in layer_ids]
            preins_ok = len(set(lv)) == len(lv)

        return plans, fixed_meetings, preins_ok, unplaceable, exhausted_pairs

    @staticmethod
    def path_nodes_of_pair(plans: dict[int, list[TimedNode]], a: int, b: int) -> set[Coord]:
        s: set[Coord] = set()
        for qid in (a, b):
            s.update(c for c, _ in plans.get(qid, []))
        return s

    @staticmethod
    def collect_layer_nodes(
        plans: dict[int, list[TimedNode]],
        fixed_meetings: dict[frozenset, Coord],
    ) -> set[Coord]:
        coords: set[Coord] = set()
        for p in plans.values():
            coords.update(c for c, _ in p)
        coords.update(fixed_meetings.values())
        return coords

    @staticmethod
    def best_meeting_candidates(
        graph: nx.Graph,
        q0: Coord,
        q1: Coord,
        reserved: set[Coord] | None = None,
        forbidden_nodes: set[Coord] | None = None,
    ) -> list[Coord]:
        reserved = reserved or set()
        forbidden_nodes = forbidden_nodes or set()

        d0 = nx.single_source_shortest_path_length(graph, q0)
        d1 = nx.single_source_shortest_path_length(graph, q1)
        ins = [n for n in graph if graph.nodes[n]["type"] == "IN" and n in d0 and n in d1]

        cands = [n for n in ins if n not in reserved and n not in forbidden_nodes]
        cands.sort(key=lambda n: (d0[n] + d1[n], max(d0[n], d1[n]), abs(d0[n] - d1[n]), n))
        return cands

    @staticmethod
    def entry_sn_from_path(path: list[TimedNode], meeting: Coord) -> Coord | None:
        first_meet_idx = None
        for i, (c, _) in enumerate(path):
            if c == meeting:
                first_meet_idx = i
                break
        if first_meet_idx is None or first_meet_idx == 0:
            return None
        return path[first_meet_idx - 1][0]

    @staticmethod
    def _preins_for_plans(
        plans: dict[int, list[TimedNode]],
        fixed_meetings: dict[frozenset, Coord],
    ) -> dict[int, Coord] | None:
        preins: dict[int, Coord] = {}
        for pair_key, meet in fixed_meetings.items():
            qids = list(pair_key)
            if len(qids) != 2:
                return None
            for qid in qids:
                path = plans.get(qid)
                if not path:
                    return None
                pin = DefaultRoutingPlanner.entry_sn_from_path(path, meet)
                if pin is None:
                    return None
                preins[qid] = pin
        return preins

    @staticmethod
    def retime_until_pre_in_wait(
        path: list[TimedNode],
        meeting: Coord,
        sync_time: int,
    ) -> list[TimedNode] | None:
        if not path:
            return None

        first_meet_idx = None
        for i, (c, _) in enumerate(path):
            if c == meeting:
                first_meet_idx = i
                break

        if first_meet_idx is None:
            return None

        if first_meet_idx == 0:
            start_node, start_t = path[0]
            new_path = [(start_node, start_t)]
            cur = start_t
            while cur < sync_time:
                new_path.append((start_node, cur + 1))
                cur += 1
            return new_path

        pre_in, t_pre = path[first_meet_idx - 1]
        new_path = path[:first_meet_idx]
        cur = t_pre
        while cur < sync_time:
            new_path.append((pre_in, cur + 1))
            cur += 1
        return new_path

    @staticmethod
    def mapf_to_targets(
        graph: nx.Graph,
        starts: dict[int, Coord],
        targets: dict[int, Coord],
        blocked_nodes: set[Coord] | None = None,
        blocked_edges: set[frozenset] | None = None,
    ) -> dict[int, list[TimedNode]]:
        if not starts:
            return {}

        res = Reservations(graph, blocked_edges=blocked_edges or set())
        blocked_nodes = blocked_nodes or set()

        allowed = set(starts.values()) | set(targets.values())
        DefaultRoutingPlanner._block_non_sn_nodes(res, graph, allowed=allowed)

        for node in blocked_nodes:
            cap = res.node_capacity(node)
            for t in range(MAX_TIME + 1):
                res.node_caps[node][t] = cap

        plans: dict[int, list[TimedNode]] = {}

        for s in starts.values():
            res.occupy_node(s, 0)

        order = sorted(
            starts.keys(),
            key=lambda q: DefaultRoutingPlanner._chebyshev(starts[q], targets[q]),
            reverse=True,
        )

        for qid in order:
            s, t = starts[qid], targets[qid]
            path = AStar.search(graph, s, t, res)
            if path is None:
                msg = f"Return routing failed for qubit {qid} from {s} -> {t}"
                raise RuntimeError(msg)
            res.commit(path)
            plans[qid] = path

        return plans

    @staticmethod
    def sample_edge_failures(
        graph: nx.Graph,
        defective_edges: set[frozenset],
        p_fail: float,
        p_repair: float,
    ) -> None:
        for u, v in graph.edges():
            e = frozenset({u, v})
            if e in defective_edges:
                if random.random() < p_repair:
                    defective_edges.discard(e)
            elif random.random() < p_fail:
                defective_edges.add(e)

    @staticmethod
    def path_uses_defective_edge(path: list[TimedNode], defective_edges: set[frozenset]) -> bool:
        return any(u != v and frozenset({u, v}) in defective_edges for (u, _), (v, _) in itertools.pairwise(path))

    @staticmethod
    def nearest_free_sn(graph: nx.Graph, source: Coord, avoid: set[Coord]) -> Coord | None:
        q = deque([source])
        seen = {source}
        while q:
            u = q.popleft()
            if graph.nodes[u]["type"] == "SN" and u not in avoid:
                return u
            for v in graph.neighbors(u):
                if v not in seen:
                    seen.add(v)
                    q.append(v)
        return None

    @staticmethod
    def _block_nodes(res: Reservations, nodes: set[Coord]) -> None:
        for node in nodes:
            cap = res.node_capacity(node)
            for t in range(MAX_TIME + 1):
                res.node_caps[node][t] = cap

    @staticmethod
    def _resolve_evacuate_collisions_with_waiters(
        graph: nx.Graph,
        evac_plans: dict[int, list[TimedNode]],
        targets: dict[int, Coord],
        current_pos: dict[int, Coord],
        waiting_qids: set[int],
        blocked_nodes: set[Coord],
        defective_edges: set[frozenset],
        blocker_to_pair: dict[int, tuple[int, int]],
    ) -> dict[int, list[TimedNode]]:
        if not evac_plans:
            return {}

        plans = dict(evac_plans)
        wait_pos: dict[int, Coord] = {qid: current_pos[qid] for qid in waiting_qids}
        targets_local: dict[int, Coord] = dict(targets)

        changed = True
        while changed:
            changed = False

            for mover_qid, mover_path in list(plans.items()):
                root_pair = blocker_to_pair.get(mover_qid)
                chain_blockers = DefaultRoutingPlanner._build_blocker_chain_along_path(
                    mover_path,
                    wait_pos=wait_pos,
                    blocker_to_pair=blocker_to_pair,
                    root_pair=root_pair,
                )
                if not chain_blockers:
                    continue

                chain_agents: list[int] = [mover_qid, *chain_blockers]
                agents_to_plan: set[int] = set(plans.keys()) | set(chain_blockers)

                starts: dict[int, Coord] = {qid: current_pos[qid] for qid in agents_to_plan}
                new_targets_full: dict[int, Coord] = {
                    qid: targets_local[qid] for qid in agents_to_plan if qid in targets_local
                }

                valid_chain = True
                for i, agent in enumerate(chain_agents):
                    if i < len(chain_agents) - 1:
                        nxt = chain_agents[i + 1]
                        new_targets_full[agent] = current_pos[nxt]
                    elif mover_qid in targets_local:
                        new_targets_full[agent] = targets_local[mover_qid]
                    else:
                        valid_chain = False
                        break

                if valid_chain and agents_to_plan.issubset(new_targets_full.keys()):
                    try:
                        replanned = DefaultRoutingPlanner.mapf_to_targets(
                            graph=graph,
                            starts=starts,
                            targets=new_targets_full,
                            blocked_nodes=blocked_nodes,
                            blocked_edges=defective_edges,
                        )
                    except RuntimeError:
                        b1 = chain_blockers[0]

                        try:
                            partial = DefaultRoutingPlanner.mapf_to_targets(
                                graph=graph,
                                starts={mover_qid: current_pos[mover_qid], b1: current_pos[b1]},
                                targets={
                                    mover_qid: current_pos[b1],
                                    b1: targets_local.get(mover_qid, current_pos[b1]),
                                },
                                blocked_nodes=blocked_nodes,
                                blocked_edges=defective_edges,
                            )
                        except RuntimeError:
                            continue

                        plans[mover_qid] = partial[mover_qid]
                        plans[b1] = partial[b1]

                        targets_local[mover_qid] = current_pos[b1]
                        targets_local[b1] = targets_local.get(mover_qid, current_pos[b1])

                        wait_pos.pop(b1, None)
                        changed = True
                        break

                    for qid, path in replanned.items():
                        plans[qid] = path
                        targets_local[qid] = new_targets_full[qid]

                    for bqid in chain_blockers:
                        wait_pos.pop(bqid, None)

                    changed = True
                    break

        return plans

    @staticmethod
    def stitch_batches(
        qubits: list[Qubit],
        batch_plans: list[dict[int, list[TimedNode]]],
        batch_defects: list[set[frozenset]] | None = None,
    ) -> tuple[dict[int, list[TimedNode]], list[tuple[int, int, set[frozenset]]]]:
        if not batch_plans:
            timelines = {q.id: [(q.pos, 0)] for q in qubits}
            return timelines, []

        durations: list[int] = []
        for plans in batch_plans:
            if not plans:
                durations.append(0)
                continue
            durations.append(max(p[-1][1] for p in plans.values()))

        initial_pos = {q.id: q.pos for q in qubits}
        timelines: dict[int, list[TimedNode]] = {q.id: [(initial_pos[q.id], 0)] for q in qubits}
        edge_timebands: list[tuple[int, int, set[frozenset]]] = []

        t_offset = 0
        for b, plans in enumerate(batch_plans):
            batch_t = durations[b]

            for q in qubits:
                qid = q.id
                last_coord, last_t = timelines[qid][-1]
                if last_t < t_offset:
                    for tt in range(last_t + 1, t_offset + 1):
                        timelines[qid].append((last_coord, tt))

            if batch_t == 0:
                continue

            for q in qubits:
                qid = q.id
                last_coord, last_t = timelines[qid][-1]

                if qid not in plans:
                    target_t = t_offset + batch_t
                    for tt in range(last_t + 1, target_t + 1):
                        timelines[qid].append((last_coord, tt))
                    continue

                local_path = plans[qid]
                shifted = [(c, t + t_offset) for (c, t) in local_path]

                if timelines[qid][-1] == shifted[0]:
                    timelines[qid].extend(shifted[1:])
                else:
                    timelines[qid].extend(shifted)

            defects = set(batch_defects[b]) if batch_defects is not None and b < len(batch_defects) else set()
            edge_timebands.append((t_offset, t_offset + batch_t, defects))

            t_offset += batch_t

        return timelines, edge_timebands
