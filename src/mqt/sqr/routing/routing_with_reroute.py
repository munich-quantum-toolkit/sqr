# Copyright (c) 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

from __future__ import annotations

import operator
from copy import deepcopy
from typing import TYPE_CHECKING

from typing_extensions import override

from mqt.sqr.routing.common import MAX_TIME, Coord, Qubit, Reservations, TimedNode
from mqt.sqr.routing.default_routing import DefaultRoutingPlanner
from mqt.sqr.routing.routing_strategy import RoutingResult, RoutingStrategy

if TYPE_CHECKING:
    import networkx as nx

MAX_REPLANS = 50
MAX_GLOBAL_ITERS = 50


class RerouteRoutingPlanner(DefaultRoutingPlanner, RoutingStrategy):
    @override
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

        layers = RerouteRoutingPlanner._build_layers_from_pairs(pairs)

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
                    batch_plans.append(RerouteRoutingPlanner._wait_plan(all_qids, current_pos, 1))
                    RerouteRoutingPlanner._snapshot_defects(batch_defects, defective_edges, 1)
                    idx += 1
                    continue

            if not fixed_meetings and not unplaceable_pairs_step1 and not exhausted_pairs_step1:
                msg = f"Layer {idx} unsolvable: no meeting-INs fixed and no spillover possible. Pairs: {layer_pairs}"
                raise RuntimeError(msg)

            if not fixed_meetings:
                batch_plans.append(RerouteRoutingPlanner._wait_plan(all_qids, current_pos, 1))
                RerouteRoutingPlanner._snapshot_defects(batch_defects, defective_edges, 1)
                idx += 1
                continue

            f_layer: set[Coord] = DefaultRoutingPlanner.collect_layer_nodes(to_meeting_plans, fixed_meetings)
            f_all = set(f_layer) | set(layer_starts)

            (
                replan_current_layer,
                to_meeting_plans,
                fixed_meetings,
                evac_plans,
                evac_targets,
                blocker_to_pair,
                blocked_nodes_evacs,
            ) = RerouteRoutingPlanner._plan_non_layer_evacuation(
                graph=graph,
                current_pos=current_pos,
                non_layer_qids=non_layer_qids,
                layer_pairs=layer_pairs,
                to_meeting_plans=to_meeting_plans,
                fixed_meetings=fixed_meetings,
                defective_edges=defective_edges,
                occupied_now=occupied_now,
                f_all=f_all,
                f_layer=f_layer,
                tried_meetings=tried_meetings,
            )

            if replan_current_layer:
                replan_counts[idx] = replan_counts.get(idx, 0) + 1
                if replan_counts[idx] > MAX_REPLANS:
                    msg = f"No valid routing for layer {idx} after {replan_counts[idx]} replans."
                    raise RuntimeError(msg)
                continue

            if not fixed_meetings:
                batch_plans.append(RerouteRoutingPlanner._wait_plan(all_qids, current_pos, 1))
                RerouteRoutingPlanner._snapshot_defects(batch_defects, defective_edges, 1)
                idx += 1
                continue

            DefaultRoutingPlanner.sample_edge_failures(
                graph, defective_edges, p_fail=(1.0 - p_success), p_repair=p_repair
            )

            (
                to_meeting_plans,
                fixed_meetings,
                evac_plans,
            ) = RerouteRoutingPlanner._reroute_or_spill_after_defects(
                graph=graph,
                current_pos=current_pos,
                layer_pairs=layer_pairs,
                defective_edges=defective_edges,
                tried_meetings=tried_meetings,
                to_meeting_plans=to_meeting_plans,
                fixed_meetings=fixed_meetings,
                evac_plans=evac_plans,
                evac_targets=evac_targets,
                blocked_nodes_evacs=blocked_nodes_evacs,
                blocker_to_pair=blocker_to_pair,
                layers=layers,
                idx=idx,
            )

            if not fixed_meetings:
                batch_plans.append(RerouteRoutingPlanner._wait_plan(all_qids, current_pos, 1))
                RerouteRoutingPlanner._snapshot_defects(batch_defects, defective_edges, 1)
                idx += 1
                continue

            _pre_in_paths, t_pre_sync = RerouteRoutingPlanner._compute_pre_in_paths(
                layer_pairs=layer_pairs,
                to_meeting_plans=to_meeting_plans,
                fixed_meetings=fixed_meetings,
                tried_meetings=tried_meetings,
            )

            if not fixed_meetings:
                batch_plans.append(RerouteRoutingPlanner._wait_plan(all_qids, current_pos, 1))
                RerouteRoutingPlanner._snapshot_defects(batch_defects, defective_edges, 1)
                idx += 1
                continue

            RerouteRoutingPlanner._execute_layer_batches(
                all_qids=all_qids,
                layer_pairs=layer_pairs,
                current_pos=current_pos,
                defective_edges=defective_edges,
                batch_plans=batch_plans,
                batch_defects=batch_defects,
                evac_plans=evac_plans,
                to_meeting_plans=to_meeting_plans,
                fixed_meetings=fixed_meetings,
                t_pre_sync=t_pre_sync,
                tried_meetings=tried_meetings,
                layers=layers,
                idx=idx,
            )

            idx += 1

        return DefaultRoutingPlanner.stitch_batches(qubits, batch_plans, batch_defects)

    @staticmethod
    def _build_layers_from_pairs(pairs: list[tuple[Qubit, Qubit]]) -> list[list[tuple[int, int]]]:
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
    def _wait_plan(
        all_qids: set[int],
        current_pos: dict[int, Coord],
        ticks: int,
    ) -> dict[int, list[TimedNode]]:
        wait: dict[int, list[TimedNode]] = {}
        for qid in all_qids:
            wait[qid] = [(current_pos[qid], 0), (current_pos[qid], ticks)]
        return wait

    @staticmethod
    def _mark_meeting_failed(
        tried_meetings: dict[frozenset, set[Coord]],
        pair: tuple[int, int],
        meet: Coord,
    ) -> None:
        key = frozenset(pair)
        tried_meetings.setdefault(key, set()).add(meet)

    @staticmethod
    def _snapshot_defects(
        batch_defects: list[set[frozenset]],
        defective_edges: set[frozenset],
        n: int,
    ) -> None:
        batch_defects.extend(set(defective_edges) for _ in range(n))

    @staticmethod
    def _plan_non_layer_evacuation(
        graph: nx.Graph,
        current_pos: dict[int, Coord],
        non_layer_qids: set[int],
        layer_pairs: list[tuple[int, int]],
        to_meeting_plans: dict[int, list[TimedNode]],
        fixed_meetings: dict[frozenset, Coord],
        defective_edges: set[frozenset],
        occupied_now: set[Coord],
        f_all: set[Coord],
        f_layer: set[Coord],
        tried_meetings: dict[frozenset, set[Coord]],
    ) -> tuple[
        bool,
        dict[int, list[TimedNode]],
        dict[frozenset, Coord],
        dict[int, list[TimedNode]],
        dict[int, Coord],
        dict[int, tuple[int, int]],
        set[Coord],
    ]:
        blockers_now: list[int] = [qid for qid in non_layer_qids if current_pos[qid] in f_all]
        if not blockers_now:
            return False, to_meeting_plans, fixed_meetings, {}, {}, {}, set()

        node_to_pairs: dict[Coord, list[tuple[int, int]]] = {}
        for a, b in layer_pairs:
            key = frozenset({a, b})
            if key not in fixed_meetings:
                continue
            nodes = DefaultRoutingPlanner.path_nodes_of_pair(to_meeting_plans, a, b)
            for n in nodes:
                node_to_pairs.setdefault(n, []).append((a, b))

        blocker_to_pair: dict[int, tuple[int, int]] = {}
        for qid in blockers_now:
            pos = current_pos[qid]
            pairs_touching = node_to_pairs.get(pos, [])
            if pairs_touching:
                blocker_to_pair[qid] = pairs_touching[0]

        avoid_for_targets = set(occupied_now) | f_all
        evac_targets: dict[int, Coord] = {}
        for qid in blockers_now:
            tgt = DefaultRoutingPlanner.nearest_free_sn(graph, current_pos[qid], avoid_for_targets)
            if tgt is not None and tgt not in f_layer:
                evac_targets[qid] = tgt
                avoid_for_targets.add(tgt)

        replan_current_layer = False
        cannot_place = [qid for qid in blockers_now if qid not in evac_targets]
        if cannot_place:
            seen_pairs: set[frozenset] = set()
            for qid in cannot_place:
                ab = blocker_to_pair.get(qid)
                if not ab:
                    continue
                pkey = frozenset(ab)
                if pkey in seen_pairs:
                    continue
                seen_pairs.add(pkey)
                meet = fixed_meetings.get(pkey)
                if meet is not None:
                    RerouteRoutingPlanner._mark_meeting_failed(tried_meetings, ab, meet)
                to_meeting_plans.pop(ab[0], None)
                to_meeting_plans.pop(ab[1], None)
                fixed_meetings.pop(pkey, None)
            replan_current_layer = True

        waiter_qids = non_layer_qids - set(evac_targets.keys())
        waiter_nodes = {current_pos[q] for q in waiter_qids}
        blocked_nodes_evacs: set[Coord] = set(f_all) | set(waiter_nodes)

        evac_plans: dict[int, list[TimedNode]] = {}
        if evac_targets and not replan_current_layer:
            try:
                evac_plans = DefaultRoutingPlanner.mapf_to_targets(
                    graph=graph,
                    starts={qid: current_pos[qid] for qid in evac_targets},
                    targets=evac_targets,
                    blocked_nodes=blocked_nodes_evacs,
                    blocked_edges=defective_edges,
                )
            except RuntimeError:
                evac_plans = {}
                for qid, tgt in evac_targets.items():
                    try:
                        one = DefaultRoutingPlanner.mapf_to_targets(
                            graph=graph,
                            starts={qid: current_pos[qid]},
                            targets={qid: tgt},
                            blocked_nodes=blocked_nodes_evacs,
                            blocked_edges=defective_edges,
                        )
                        evac_plans[qid] = one[qid]
                    except RuntimeError:
                        ab = blocker_to_pair.get(qid)
                        if ab:
                            pkey = frozenset(ab)
                            meet = fixed_meetings.get(pkey)
                            if meet is not None:
                                RerouteRoutingPlanner._mark_meeting_failed(tried_meetings, ab, meet)
                            to_meeting_plans.pop(ab[0], None)
                            to_meeting_plans.pop(ab[1], None)
                            fixed_meetings.pop(pkey, None)
                            replan_current_layer = True

        if evac_plans and not replan_current_layer:
            waiting_qids = non_layer_qids - set(evac_targets.keys())
            evac_plans = RerouteRoutingPlanner._resolve_evacuate_collisions_with_waiters(
                graph=graph,
                evac_plans=evac_plans,
                targets=evac_targets,
                current_pos=current_pos,
                waiting_qids=waiting_qids,
                blocked_nodes=blocked_nodes_evacs,
                defective_edges=defective_edges,
                blocker_to_pair=blocker_to_pair,
            )

        return (
            replan_current_layer,
            to_meeting_plans,
            fixed_meetings,
            evac_plans,
            evac_targets,
            blocker_to_pair,
            blocked_nodes_evacs,
        )

    @staticmethod
    def _reroute_or_spill_after_defects(
        graph: nx.Graph,
        current_pos: dict[int, Coord],
        layer_pairs: list[tuple[int, int]],
        defective_edges: set[frozenset],
        tried_meetings: dict[frozenset, set[Coord]],
        to_meeting_plans: dict[int, list[TimedNode]],
        fixed_meetings: dict[frozenset, Coord],
        evac_plans: dict[int, list[TimedNode]],
        evac_targets: dict[int, Coord],
        blocked_nodes_evacs: set[Coord],
        blocker_to_pair: dict[int, tuple[int, int]],
        layers: list[list[tuple[int, int]]],
        idx: int,
    ) -> tuple[dict[int, list[TimedNode]], dict[frozenset, Coord], dict[int, list[TimedNode]]]:
        if evac_plans:
            broken_movers = {
                qid
                for qid, path in evac_plans.items()
                if DefaultRoutingPlanner.path_uses_defective_edge(path, defective_edges)
            }
            if broken_movers:
                try:
                    evac_plans = DefaultRoutingPlanner.mapf_to_targets(
                        graph=graph,
                        starts={qid: current_pos[qid] for qid in evac_plans},
                        targets={qid: evac_targets[qid] for qid in evac_plans},
                        blocked_nodes=blocked_nodes_evacs,
                        blocked_edges=defective_edges,
                    )
                except RuntimeError:
                    to_spill_for_nonlayer: set[tuple[int, int]] = set()
                    for qid in broken_movers:
                        ab = blocker_to_pair.get(qid)
                        if ab:
                            to_spill_for_nonlayer.add(ab)

                    if to_spill_for_nonlayer:
                        for a, b in to_spill_for_nonlayer:
                            key = frozenset({a, b})
                            meet = fixed_meetings.get(key)
                            if meet is not None:
                                RerouteRoutingPlanner._mark_meeting_failed(tried_meetings, (a, b), meet)
                            to_meeting_plans.pop(a, None)
                            to_meeting_plans.pop(b, None)
                            fixed_meetings.pop(key, None)

                        layers[idx + 1 : idx + 1] = [list(to_spill_for_nonlayer)]

                    evac_plans.clear()

        if not fixed_meetings:
            return to_meeting_plans, fixed_meetings, evac_plans

        need_reroute_pairs = RerouteRoutingPlanner._pairs_needing_reroute(
            layer_pairs=layer_pairs,
            fixed_meetings=fixed_meetings,
            to_meeting_plans=to_meeting_plans,
            defective_edges=defective_edges,
        )

        if need_reroute_pairs:
            ok_local, local_plans = RerouteRoutingPlanner._try_local_triangle_bypass_for_pairs(
                graph=graph,
                current_pos=current_pos,
                fixed_meetings=fixed_meetings,
                keep_pairs=need_reroute_pairs,
                defective_edges=defective_edges,
                existing_layer_plans=to_meeting_plans,
                existing_evac_plans=evac_plans,
            )
            to_meeting_plans.update(local_plans)

            not_ok = [ab for ab in need_reroute_pairs if ab not in ok_local]
            if not_ok:
                for a, b in not_ok:
                    key = frozenset({a, b})
                    meet = fixed_meetings.get(key)
                    if meet is not None:
                        RerouteRoutingPlanner._mark_meeting_failed(tried_meetings, (a, b), meet)
                    to_meeting_plans.pop(a, None)
                    to_meeting_plans.pop(b, None)
                    fixed_meetings.pop(key, None)
                layers[idx + 1 : idx + 1] = [not_ok]

        return to_meeting_plans, fixed_meetings, evac_plans

    @staticmethod
    def _pairs_needing_reroute(
        layer_pairs: list[tuple[int, int]],
        fixed_meetings: dict[frozenset, Coord],
        to_meeting_plans: dict[int, list[TimedNode]],
        defective_edges: set[frozenset],
    ) -> set[tuple[int, int]]:
        need: set[tuple[int, int]] = set()

        for a, b in layer_pairs:
            key = frozenset({a, b})
            if key not in fixed_meetings:
                continue
            meet = fixed_meetings[key]

            cut_a = DefaultRoutingPlanner.retime_until_pre_in_wait(to_meeting_plans[a], meet, 0)
            cut_b = DefaultRoutingPlanner.retime_until_pre_in_wait(to_meeting_plans[b], meet, 0)
            if cut_a is None or cut_b is None:
                need.add((a, b))
                continue

            if DefaultRoutingPlanner.path_uses_defective_edge(cut_a, defective_edges):
                need.add((a, b))
                continue
            if DefaultRoutingPlanner.path_uses_defective_edge(cut_b, defective_edges):
                need.add((a, b))
                continue

            pre_a = cut_a[-1][0]
            pre_b = cut_b[-1][0]
            if frozenset({pre_a, meet}) in defective_edges or frozenset({pre_b, meet}) in defective_edges:
                need.add((a, b))

        return need

    @staticmethod
    def _compute_pre_in_paths(
        layer_pairs: list[tuple[int, int]],
        to_meeting_plans: dict[int, list[TimedNode]],
        fixed_meetings: dict[frozenset, Coord],
        tried_meetings: dict[frozenset, set[Coord]],
    ) -> tuple[dict[int, list[TimedNode]], int]:
        pre_in_paths: dict[int, list[TimedNode]] = {}
        t_pre_sync = 0

        for a, b in layer_pairs:
            key = frozenset({a, b})
            if key not in fixed_meetings:
                continue

            meet = fixed_meetings[key]
            pa = DefaultRoutingPlanner.retime_until_pre_in_wait(to_meeting_plans[a], meet, 0)
            pb = DefaultRoutingPlanner.retime_until_pre_in_wait(to_meeting_plans[b], meet, 0)
            if pa is None or pb is None:
                RerouteRoutingPlanner._mark_meeting_failed(tried_meetings, (a, b), meet)
                to_meeting_plans.pop(a, None)
                to_meeting_plans.pop(b, None)
                fixed_meetings.pop(key, None)
                continue

            pre_in_paths[a] = pa
            pre_in_paths[b] = pb
            if pa:
                t_pre_sync = max(t_pre_sync, pa[-1][1])
            if pb:
                t_pre_sync = max(t_pre_sync, pb[-1][1])

        return pre_in_paths, t_pre_sync

    @staticmethod
    def _execute_layer_batches(
        all_qids: set[int],
        layer_pairs: list[tuple[int, int]],
        current_pos: dict[int, Coord],
        defective_edges: set[frozenset],
        batch_plans: list[dict[int, list[TimedNode]]],
        batch_defects: list[set[frozenset]],
        evac_plans: dict[int, list[TimedNode]],
        to_meeting_plans: dict[int, list[TimedNode]],
        fixed_meetings: dict[frozenset, Coord],
        t_pre_sync: int,
        tried_meetings: dict[frozenset, set[Coord]],
        layers: list[list[tuple[int, int]]],
        idx: int,
    ) -> None:
        if evac_plans and any(
            DefaultRoutingPlanner.path_uses_defective_edge(path, defective_edges) for path in evac_plans.values()
        ):
            evac_plans.clear()

        if evac_plans:
            micro_evacuate: dict[int, list[TimedNode]] = dict(evac_plans.items())
            dur = max((p[-1][1] for p in micro_evacuate.values()), default=0)

            for qid in all_qids - set(micro_evacuate.keys()):
                micro_evacuate[qid] = [(current_pos[qid], 0), (current_pos[qid], dur)]

            batch_plans.append(micro_evacuate)
            RerouteRoutingPlanner._snapshot_defects(batch_defects, defective_edges, 1)

            for qid in evac_plans:
                current_pos[qid] = evac_plans[qid][-1][0]

        micro_to_pre: dict[int, list[TimedNode]] = {}
        exec_layer_qids: set[int] = set()

        for a, b in layer_pairs:
            key = frozenset({a, b})
            if key not in fixed_meetings:
                continue

            meet = fixed_meetings[key]
            cut_a = DefaultRoutingPlanner.retime_until_pre_in_wait(to_meeting_plans[a], meet, t_pre_sync)
            cut_b = DefaultRoutingPlanner.retime_until_pre_in_wait(to_meeting_plans[b], meet, t_pre_sync)

            if cut_a is None or cut_b is None:
                RerouteRoutingPlanner._mark_meeting_failed(tried_meetings, (a, b), meet)
                fixed_meetings.pop(key, None)
                to_meeting_plans.pop(a, None)
                to_meeting_plans.pop(b, None)
                continue

            micro_to_pre[a] = cut_a
            micro_to_pre[b] = cut_b
            exec_layer_qids.update({a, b})

        dur_pre = max((p[-1][1] for p in micro_to_pre.values()), default=0)
        for qid in all_qids - exec_layer_qids:
            micro_to_pre[qid] = [(current_pos[qid], 0), (current_pos[qid], dur_pre)]

        batch_plans.append(micro_to_pre)
        RerouteRoutingPlanner._snapshot_defects(batch_defects, defective_edges, 1)

        for qid in exec_layer_qids:
            current_pos[qid] = micro_to_pre[qid][-1][0]

        micro_in: dict[int, list[TimedNode]] = {}
        spill_after_micro: list[tuple[int, int]] = []

        for a, b in layer_pairs:
            key = frozenset({a, b})
            if key not in fixed_meetings:
                continue

            meet = fixed_meetings[key]
            pre_a = current_pos[a]
            pre_b = current_pos[b]

            if frozenset({pre_a, meet}) in defective_edges or frozenset({pre_b, meet}) in defective_edges:
                spill_after_micro.append((a, b))
                micro_in[a] = [(pre_a, 0), (pre_a, 2)]
                micro_in[b] = [(pre_b, 0), (pre_b, 2)]
            else:
                micro_in[a] = [(pre_a, 0), (meet, 1), (pre_a, 2)]
                micro_in[b] = [(pre_b, 0), (meet, 1), (pre_b, 2)]

        active_layer_qids = {q for ab in layer_pairs if frozenset(ab) in fixed_meetings for q in ab}
        for qid in all_qids - active_layer_qids:
            micro_in[qid] = [(current_pos[qid], 0), (current_pos[qid], 2)]

        batch_plans.append(micro_in)
        RerouteRoutingPlanner._snapshot_defects(batch_defects, defective_edges, 1)

        if spill_after_micro:
            for a, b in spill_after_micro:
                key = frozenset({a, b})
                meet = fixed_meetings.get(key)
                if meet is not None:
                    RerouteRoutingPlanner._mark_meeting_failed(tried_meetings, (a, b), meet)
                fixed_meetings.pop(key, None)

            layers[idx + 1 : idx + 1] = [spill_after_micro]

        for a, b in [tuple(sorted(k)) for k in fixed_meetings]:
            tried_meetings.pop(frozenset({a, b}), None)

    @staticmethod
    def _reserve_existing_plans(
        res: Reservations,
        plans: dict[int, list[TimedNode]],
        skip_qids: set[int] | None = None,
    ) -> None:
        skip_qids = skip_qids or set()
        for qid, path in plans.items():
            if qid in skip_qids:
                continue
            Reservations.commit(res, path)

    @staticmethod
    def _common_triangle_candidates(
        graph: nx.Graph,
        u: Coord,
        v: Coord,
        banned_nodes: set[Coord] | None = None,
    ) -> list[Coord]:
        banned_nodes = banned_nodes or set()
        neighbors_u = set(graph.neighbors(u))
        neighbors_v = set(graph.neighbors(v))
        cand = list(neighbors_u & neighbors_v)
        cand = [w for w in cand if w not in {u, v} and w not in banned_nodes]
        cand.sort(key=operator.itemgetter(0, 1))
        return cand

    @staticmethod
    def _patch_path_with_triangle_bypass(
        graph: nx.Graph,
        path: list[TimedNode],
        defective_edges: set[frozenset],
        blocked_nodes_static: set[Coord] | None = None,
    ) -> list[TimedNode]:
        if not path or len(path) < 2:
            return path

        blocked_nodes_static = blocked_nodes_static or set()

        new_path: list[TimedNode] = [path[0]]
        total_extra = 0

        for i in range(1, len(path)):
            u, tu = new_path[-1]
            v, tv_orig = path[i]
            tv = tv_orig + total_extra

            if frozenset({u, v}) not in defective_edges:
                new_path.append((v, tv))
                continue

            candidates = RerouteRoutingPlanner._common_triangle_candidates(
                graph, u, v, banned_nodes=blocked_nodes_static
            )

            picked_w: Coord | None = None
            for w in candidates:
                if frozenset({u, w}) in defective_edges:
                    continue
                if frozenset({w, v}) in defective_edges:
                    continue
                picked_w = w
                break

            if picked_w is None:
                new_path.append((v, tv))
                continue

            new_path.extend(((picked_w, tu + 1), (v, tu + 2)))
            total_extra += 1

        return new_path

    @staticmethod
    def _try_local_triangle_bypass_for_pairs(
        graph: nx.Graph,
        current_pos: dict[int, Coord],
        fixed_meetings: dict[frozenset, Coord],
        keep_pairs: set[tuple[int, int]],
        defective_edges: set[frozenset],
        existing_layer_plans: dict[int, list[TimedNode]] | None,
        existing_evac_plans: dict[int, list[TimedNode]] | None,
    ) -> tuple[set[tuple[int, int]], dict[int, list[TimedNode]]]:
        ok_pairs: set[tuple[int, int]] = set()
        new_plans: dict[int, list[TimedNode]] = {}

        res_base = Reservations(graph, blocked_edges=defective_edges)

        skip_qids: set[int] = {q for ab in keep_pairs for q in ab}
        if existing_layer_plans:
            RerouteRoutingPlanner._reserve_existing_plans(res_base, existing_layer_plans, skip_qids=skip_qids)
        if existing_evac_plans:
            RerouteRoutingPlanner._reserve_existing_plans(res_base, existing_evac_plans, skip_qids=None)

        moving_qids: set[int] = set()
        if existing_layer_plans:
            moving_qids |= set(existing_layer_plans.keys())
        if existing_evac_plans:
            moving_qids |= set(existing_evac_plans.keys())

        stationary_nodes: set[Coord] = {
            current_pos[qid] for qid in current_pos if (qid not in moving_qids) and (qid not in skip_qids)
        }
        for node in stationary_nodes:
            cap = res_base.node_capacity(node)
            for t in range(MAX_TIME + 1):
                res_base.node_caps[node][t] = cap
        for node in stationary_nodes:
            Reservations.commit(res_base, [(node, 0), (node, MAX_TIME)])

        placed_preins: set[Coord] = set()
        if existing_layer_plans:
            for pair_key, meet in fixed_meetings.items():
                a, b = list(pair_key)
                if (a, b) in keep_pairs or (b, a) in keep_pairs:
                    continue
                for qid in (a, b):
                    pth = existing_layer_plans.get(qid)
                    if not pth:
                        continue
                    pin = DefaultRoutingPlanner.entry_sn_from_path(pth, meet)
                    if pin is not None:
                        placed_preins.add(pin)
            for pin in placed_preins:
                cap = res_base.node_capacity(pin)
                for t in range(MAX_TIME + 1):
                    res_base.node_caps[pin][t] = cap

        evac_target_nodes: set[Coord] = set()
        if existing_evac_plans:
            for p in existing_evac_plans.values():
                if p:
                    evac_target_nodes.add(p[-1][0])
        for node in evac_target_nodes:
            cap = res_base.node_capacity(node)
            for t in range(MAX_TIME + 1):
                res_base.node_caps[node][t] = cap

        blocked_nodes_static: set[Coord] = set(stationary_nodes)

        def md(a: Coord, b: Coord) -> int:
            return abs(a[0] - b[0]) + abs(a[1] - b[1])

        order = sorted(
            keep_pairs,
            key=lambda ab: (
                md(current_pos[ab[0]], fixed_meetings[frozenset(ab)])
                + md(current_pos[ab[1]], fixed_meetings[frozenset(ab)])
            ),
        )

        for a, b in order:
            meet = fixed_meetings[frozenset({a, b})]
            pa = existing_layer_plans.get(a) if existing_layer_plans else None
            pb = existing_layer_plans.get(b) if existing_layer_plans else None
            if pa is None or pb is None:
                continue

            uses_defect_a = DefaultRoutingPlanner.path_uses_defective_edge(pa, defective_edges)
            uses_defect_b = DefaultRoutingPlanner.path_uses_defective_edge(pb, defective_edges)
            if not (uses_defect_a or uses_defect_b):
                continue

            pa_patched = (
                RerouteRoutingPlanner._patch_path_with_triangle_bypass(
                    graph, pa, defective_edges, blocked_nodes_static=blocked_nodes_static
                )
                if uses_defect_a
                else pa
            )
            pb_patched = (
                RerouteRoutingPlanner._patch_path_with_triangle_bypass(
                    graph, pb, defective_edges, blocked_nodes_static=blocked_nodes_static
                )
                if uses_defect_b
                else pb
            )

            pre_a = DefaultRoutingPlanner.entry_sn_from_path(pa_patched, meet)
            pre_b = DefaultRoutingPlanner.entry_sn_from_path(pb_patched, meet)
            if pre_a is None or pre_b is None:
                continue
            if pre_a == pre_b:
                continue
            if pre_a in placed_preins or pre_b in placed_preins:
                continue

            res_try = deepcopy(res_base)
            try:
                Reservations.commit(res_try, pa_patched)
                Reservations.commit(res_try, pb_patched)
            except IndexError:
                continue

            if DefaultRoutingPlanner.path_uses_defective_edge(pa_patched, defective_edges):
                continue
            if DefaultRoutingPlanner.path_uses_defective_edge(pb_patched, defective_edges):
                continue

            Reservations.commit(res_base, pa_patched)
            Reservations.commit(res_base, pb_patched)

            placed_preins.update({pre_a, pre_b})
            new_plans[a] = pa_patched
            new_plans[b] = pb_patched
            ok_pairs.add((a, b))

        return ok_pairs, new_plans
