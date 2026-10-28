"""Deterministic temporal workloads; oracle labels come from an event log, never an LLM."""

import random
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Literal

from .models import Probe, Scenario, Suite, Update

PEOPLE = [
    "Avery Nolan",
    "Blair Okafor",
    "Casey Moreno",
    "Devon Shah",
    "Ellis Romano",
    "Finley Park",
    "Greer Costa",
    "Harper Bell",
    "Indigo Reyes",
    "Jules Mensah",
    "Kai Laurent",
    "Lane Patel",
    "Morgan Sato",
    "Noel Haddad",
    "Oakley Chen",
    "Parker Silva",
    "Quinn Ibrahim",
    "Reese Novak",
    "Sage Wallace",
    "Taylor Kim",
    "Uma Fischer",
    "Val Rivera",
    "Winter Brooks",
    "Yael Martin",
    "Zion Ahmed",
    "Arden Lee",
    "Bryn Hughes",
    "Cameron Rossi",
    "Drew Khan",
    "Emery Lewis",
    "Frankie Reed",
    "Gray Tan",
    "Hayden Ali",
    "Imani Cruz",
    "Jordan Fox",
    "Kendall Wu",
    "Logan Price",
    "Marin Cole",
    "Nico Stone",
    "Owen Blake",
]


@dataclass(frozen=True)
class Assignment:
    effective_at: datetime
    owner: str
    revision: int


def owner_at(events: list[Assignment], at: datetime) -> str:
    eligible = [e for e in events if e.effective_at <= at]
    if not eligible:
        raise ValueError("No assignment known at the requested time.")
    return max(eligible, key=lambda e: e.effective_at).owner


def temporal_stress_suite(
    depths: tuple[int, ...] = (4, 12, 32),
    seeds: tuple[int, ...] = (7, 23, 41),
    orders: tuple[Literal["chronological", "late"], ...] = ("chronological", "late"),
    budgets: tuple[int, ...] = (512, 2048),
) -> Suite:
    if not depths or any(d < 3 or d > len(PEOPLE) for d in depths):
        raise ValueError(f"History depth must be between 3 and {len(PEOPLE)}.")
    if not seeds or not orders or any(o not in {"chronological", "late"} for o in orders):
        raise ValueError("Select seeds and chronological/late orders.")
    if not budgets or any(b < 128 or b > 16000 for b in budgets):
        raise ValueError("Evidence budgets must be between 128 and 16000 tokens.")
    for values in [depths, seeds, orders, budgets]:
        if len(values) != len(set(values)):
            raise ValueError("Factor values must be distinct.")
    scenarios = []
    for depth in depths:
        for seed in seeds:
            # Paired order variants use exactly the same facts, people, wording and queries.
            rng = random.Random(seed)
            people = rng.sample(PEOPLE, depth)
            project = f"Meridian-{seed}"
            base = datetime(2025, 1, 1, tzinfo=UTC)
            events = [
                Assignment(base + timedelta(days=i * 7), name, i) for i, name in enumerate(people)
            ]
            for order in orders:
                stream = list(events)
                if order == "late":
                    # Initial and latest events are seen before delayed middle revisions.
                    middle = list(events[1:-1])
                    random.Random(seed + 1000).shuffle(middle)
                    stream = [events[0], events[-1], *middle]
                steps = []
                # Similar project and a different role are stable distractors; neither changes
                # the oracle ownership timeline. Source documents do not contain oracle labels.
                steps.append(
                    Update(
                        id="reviewer",
                        document_id="reviewer",
                        effective_at=base,
                        text=f"The security reviewer for Project {project} is Rowan Bennett.",
                    )
                )
                steps.append(
                    Update(
                        id="similar-project",
                        document_id="similar-project",
                        effective_at=base,
                        text=f"Project {project}-Archive has sole accountable owner Rowan Bennett. "
                        f"Project {project}-Archive is distinct from Project {project}.",
                    )
                )
                templates = [
                    "{name} is now the sole accountable owner of Project {project}.",
                    "Project {project} has a new sole accountable owner: {name}.",
                    "The sole accountable ownership of Project {project} transfers to {name}.",
                ]
                for event in stream:
                    steps.append(
                        Update(
                            id=f"owner-{event.revision:02d}",
                            document_id="ownership-register",
                            effective_at=event.effective_at,
                            text=templates[event.revision % len(templates)].format(
                                name=event.owner, project=project
                            ),
                        )
                    )
                    if event.revision % 4 == 0:
                        steps.append(
                            Update(
                                id=f"audit-{event.revision:02d}",
                                document_id=f"independent-audit-{event.revision:02d}",
                                effective_at=event.effective_at,
                                text=f"The independent operations audit confirms that {event.owner} "
                                f"is the sole accountable owner of Project {project}.",
                            )
                        )
                # Ask after every revision has arrived, so order variants share gold answers.
                # No source sentence explicitly says who was superseded or when their tenure ends.
                times = [
                    ("current", events[-1].effective_at + timedelta(days=1)),
                    ("early", events[0].effective_at + timedelta(days=1)),
                    ("middle", events[depth // 2].effective_at + timedelta(days=1)),
                    # Exact boundary: new ownership applies from this instant, not the old one.
                    ("boundary", events[-1].effective_at),
                ]
                probes = []
                for label, at in times:
                    expected = owner_at(events, at)
                    for budget in budgets:
                        probes.append(
                            Probe(
                                id=f"{label}-b{budget}",
                                query=f"Who is the sole accountable owner of Project {project} "
                                f"as of {at.date().isoformat()}? Give only the person's full name.",
                                expected_any=[expected],
                                forbidden_any=[name for name in people if name != expected]
                                + ["Rowan Bennett"],
                                as_of=at,
                                category="current" if label == "current" else "historical",
                                dimension="temporal",
                                temporal_role="current" if label == "current" else "historical",
                                context_tokens=budget,
                            )
                        )
                steps[-1].probes = probes
                scenarios.append(
                    Scenario(
                        id=f"history-{depth:02d}-seed-{seed}-{order}",
                        description=f"{depth} successive owners, {order} arrival, stable reviewer and "
                        "similar-project distractors; no explicit end dates in source facts.",
                        metadata={
                            "family": "ownership-churn",
                            "history_depth": depth,
                            "seed": seed,
                            "arrival_order": order,
                        },
                        steps=steps,
                    )
                )
    return Suite(
        name="Temporal stress v1",
        description="Controlled history-depth and arrival-order experiments. The same relational "
        "facts, questions, oracle and evidence budgets are used for every backend. "
        "A harder workload is not a promise that Graphiti wins.",
        metadata={
            "kind": "temporal-stress",
            "generator_version": 1,
            "depths": list(depths),
            "seeds": list(seeds),
            "orders": list(orders),
            "context_budgets": list(budgets),
            "overlapping_sources": True,
            "oracle": "latest effective event at query time",
            "limitations": [
                "Synthetic ownership workload; not a comprehensive application benchmark.",
                "Paired probes within a scenario are correlated.",
                "Does not test retroactive corrections, provenance trust ranking or multi-hop joins.",
            ],
        },
        scenarios=scenarios,
    )
