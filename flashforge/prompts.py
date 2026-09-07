"""Domain-tagged prompt set for routing traces.

Domain labels exist for Q4 (does expert usage cluster by domain?). The test is
only meaningful if the domains are genuinely different kinds of text, so these
lean hard on register and vocabulary rather than topic alone.

These are a starting set, sized to give a few thousand tokens total. For
publishable numbers, point --prompts at a JSONL of real corpus text
({"id":..., "domain":..., "text":...} per line) — a couple of hundred
sequences of 512+ tokens each.
"""

from __future__ import annotations

import json
from pathlib import Path

DEFAULT_PROMPTS: list[dict[str, str]] = [
    # ---- code ----
    {"domain": "code", "text": "Here is a Python implementation of a least-recently-used cache with an O(1) get and put, built on a doubly linked list and a dictionary. The node class stores a key, a value, and previous and next pointers. The cache keeps a sentinel head and tail so that eviction never has to special-case an empty list. On get, we unlink the node and re-append it next to the head; on put, we either update in place or insert and then evict from the tail when capacity is exceeded."},
    {"domain": "code", "text": "def quicksort(items, lo=0, hi=None):\n    if hi is None:\n        hi = len(items) - 1\n    if lo >= hi:\n        return items\n    pivot = items[(lo + hi) // 2]\n    left, right = lo, hi\n    while left <= right:\n        while items[left] < pivot:\n            left += 1\n        while items[right] > pivot:\n            right -= 1\n        if left <= right:\n            items[left], items[right] = items[right], items[left]\n            left += 1\n            right -= 1\n    quicksort(items, lo, right)\n    quicksort(items, left, hi)\n    return items"},
    {"domain": "code", "text": "The CUDA kernel launches one thread block per row of the output matrix. Each block cooperatively loads a tile of the input into shared memory, synchronises, and then every thread accumulates a partial dot product across the tile. Because shared memory is banked, the tile is padded by one column to avoid bank conflicts on the strided read. After the accumulation loop the partial sums are reduced within the warp using shuffle instructions, and lane zero writes the result back to global memory with a coalesced store."},
    {"domain": "code", "text": "When the migration runs, it first acquires an advisory lock so that two application instances cannot apply the same schema change concurrently. It then checks the schema_migrations table for the highest applied version, computes the set of pending migration files, and applies them inside a single transaction. If any statement fails the whole transaction rolls back and the advisory lock is released in a finally block, leaving the database in its previous consistent state."},

    # ---- math ----
    {"domain": "math", "text": "Let f be a continuous function on the closed interval [a, b] and differentiable on the open interval (a, b). The mean value theorem asserts that there exists a point c in (a, b) such that f'(c) equals the average rate of change of f across the interval, that is, the quotient of f(b) minus f(a) by b minus a. The proof proceeds by applying Rolle's theorem to the auxiliary function formed by subtracting from f the secant line through the two endpoints."},
    {"domain": "math", "text": "Consider a Markov chain on a finite state space with transition matrix P. If the chain is irreducible and aperiodic then it possesses a unique stationary distribution pi satisfying pi P equals pi, and the distribution of the chain at time n converges to pi geometrically fast in total variation distance. The rate of convergence is governed by the spectral gap, the difference between the largest eigenvalue, which equals one, and the modulus of the second largest eigenvalue."},
    {"domain": "math", "text": "To compute the integral of x squared times e to the negative x from zero to infinity, integrate by parts twice. Taking u equal to x squared and dv equal to e to the negative x dx yields a boundary term that vanishes at both limits, leaving twice the integral of x times e to the negative x. Repeating the procedure reduces this to twice the integral of e to the negative x, which evaluates to two. The result is the gamma function evaluated at three, namely factorial of two."},
    {"domain": "math", "text": "The singular value decomposition factors any real matrix A of shape m by n into the product U S V transpose, where U and V are orthogonal and S is diagonal with non-negative entries in descending order. The columns of V are eigenvectors of A transpose A, the columns of U are eigenvectors of A A transpose, and the singular values are the square roots of the shared non-zero eigenvalues. Truncating the decomposition after k terms gives the best rank-k approximation in the Frobenius norm."},

    # ---- prose ----
    {"domain": "prose", "text": "The rain had been falling since before dawn, and by the time the light came up the yard was a grey sheet of standing water with the fence posts rising out of it like the masts of a sunken fleet. She stood at the kitchen window with her hands around a cup that had gone cold an hour ago and watched the water creep toward the step. There was nothing to be done about it now. The gutters had been her husband's job, and he had been three years gone."},
    {"domain": "prose", "text": "He remembered the house as enormous, its corridors endless, the garden a wilderness that could swallow an afternoon. Returning at forty he found a narrow terrace on a street of identical terraces, the garden a strip of grass he could cross in nine paces. Nothing had shrunk, of course. He had simply grown into a person for whom the world was smaller, and he could not decide whether this was a loss or merely the ordinary arithmetic of getting older."},
    {"domain": "prose", "text": "The market opened at four in the morning and by five the whole street smelled of diesel and crushed mint. Vendors shouted across the aisles in a shorthand built from decades of proximity, half of it insult and half of it affection, none of it intelligible to anyone who had arrived in the last twenty years. The old woman at the corner stall had not changed her prices since the currency was reissued, and she would not be persuaded to."},
    {"domain": "prose", "text": "Winter arrived that year without ceremony. One evening the trees still held their leaves and the next morning the branches were bare and black against a sky the colour of wet paper. The village settled into its long quiet, the streets emptying by four, the windows filling with the small yellow light of rooms where people had decided to stay put until spring. Even the dogs seemed to have agreed to it."},

    # ---- factual ----
    {"domain": "factual", "text": "Photosynthesis in higher plants occurs in two coupled stages. The light-dependent reactions take place in the thylakoid membranes, where chlorophyll absorbs photons and drives the transfer of electrons along a chain of carriers, generating ATP and NADPH while splitting water and releasing oxygen. The light-independent reactions, occurring in the stroma, use that ATP and NADPH to fix carbon dioxide into three-carbon sugars through the Calvin cycle, with the enzyme RuBisCO catalysing the initial carboxylation step."},
    {"domain": "factual", "text": "The Antikythera mechanism is a geared bronze device recovered from a Roman-era shipwreck off the Greek island of the same name in 1901. Dated to roughly the second century BCE, it modelled the positions of the sun and moon and predicted eclipses using a train of at least thirty interlocking bronze gears, including a differential arrangement to represent the moon's variable speed. Nothing of comparable mechanical sophistication is known from the following fourteen centuries."},
    {"domain": "factual", "text": "Plate tectonics describes the outer shell of the Earth as a set of rigid plates moving over a ductile asthenosphere. New oceanic crust forms at mid-ocean ridges where plates diverge and magma rises to fill the gap, and is destroyed at subduction zones where denser oceanic lithosphere descends beneath continental margins. The resulting cycle explains the distribution of earthquakes, volcanic arcs, mountain belts, and the observed symmetry of magnetic striping on either side of spreading centres."},
    {"domain": "factual", "text": "The printing press that Gutenberg assembled around 1450 combined several existing technologies: the screw press used in winemaking, oil-based inks adapted from painting, and above all a hand mould that allowed individual metal type to be cast quickly and to a consistent height. It was the mould, rather than the press itself, that made the system economic, since a single punch could produce effectively unlimited identical sorts of a given letter."},

    # ---- dialogue ----
    {"domain": "dialogue", "text": "\"You're telling me it was working yesterday.\"\n\"It was working yesterday.\"\n\"And nothing changed.\"\n\"Nothing changed.\"\n\"Then explain to me why the logs stop at eleven forty.\"\n\"I can't.\"\n\"Try.\"\n\"I genuinely can't. I looked at the deploy history, I looked at the config, I looked at the upstream. If something changed, it changed somewhere I don't have visibility into.\"\n\"That's not an answer, that's a shrug with extra steps.\"\n\"It's the honest version of a shrug. Would you prefer I made something up?\""},
    {"domain": "dialogue", "text": "\"Do you want the good news or the bad news?\"\n\"Is the good news actually good, or is it just the bad news wearing a hat?\"\n\"It's the second one.\"\n\"Then skip it.\"\n\"Fine. We lost the account.\"\n\"How badly?\"\n\"All of it. Effective end of month.\"\n\"And the part you were going to dress up as good news?\"\n\"They paid the outstanding invoices first. All four of them, same morning they sent the termination notice. Somebody over there has a conscience.\""},
    {"domain": "dialogue", "text": "\"How long have you been sitting here?\"\n\"What time is it?\"\n\"Nearly two.\"\n\"Then about five hours.\"\n\"Have you eaten?\"\n\"There was a thing. Earlier. Some kind of pastry.\"\n\"That's not eating, that's grazing.\"\n\"It had a filling.\"\n\"Get your coat. There's a place on the corner that stays open, and you're going to order something with a vegetable in it, and then you're going to go home and sleep, and the code will still be broken in the morning.\""},
    {"domain": "dialogue", "text": "\"I want to be clear that I'm not agreeing with you.\"\n\"Noted.\"\n\"I'm going along with it because we've run out of time to argue, not because you've convinced me.\"\n\"Also noted.\"\n\"And if it goes wrong I'm going to be unbearable about it.\"\n\"I'd expect nothing less. Can we start now?\"\n\"We can start now.\""},

    # ---- structured ----
    {"domain": "structured", "text": "{\"order_id\": \"A-88213\", \"status\": \"shipped\", \"customer\": {\"id\": 4471, \"tier\": \"gold\", \"region\": \"EMEA\"}, \"items\": [{\"sku\": \"KB-104\", \"qty\": 1, \"unit_price\": 89.00}, {\"sku\": \"MS-220\", \"qty\": 2, \"unit_price\": 34.50}], \"totals\": {\"subtotal\": 158.00, \"tax\": 31.60, \"shipping\": 0.00, \"grand_total\": 189.60}, \"shipped_at\": \"2026-03-14T09:22:11Z\", \"carrier\": \"DHL\", \"tracking\": \"JD0141900012345678\"}"},
    {"domain": "structured", "text": "CREATE TABLE expert_access (\n  trace_id   BIGINT NOT NULL,\n  seq_id     INTEGER NOT NULL,\n  pos        INTEGER NOT NULL,\n  layer      SMALLINT NOT NULL,\n  expert     SMALLINT NOT NULL,\n  gate_weight REAL NOT NULL,\n  PRIMARY KEY (trace_id, seq_id, pos, layer, expert)\n);\nCREATE INDEX idx_expert_access_layer_expert ON expert_access (layer, expert);\nCREATE INDEX idx_expert_access_seq_pos ON expert_access (seq_id, pos);\nANALYZE expert_access;"},
    {"domain": "structured", "text": "version: \"3.9\"\nservices:\n  api:\n    image: registry.internal/api:1.14.2\n    ports:\n      - \"8080:8080\"\n    environment:\n      DATABASE_URL: postgres://api:secret@db:5432/api\n      LOG_LEVEL: info\n    depends_on:\n      db:\n        condition: service_healthy\n  db:\n    image: postgres:16-alpine\n    volumes:\n      - pgdata:/var/lib/postgresql/data\n    healthcheck:\n      test: [\"CMD-SHELL\", \"pg_isready -U api\"]\n      interval: 5s\n      retries: 10\nvolumes:\n  pgdata:"},
    {"domain": "structured", "text": "| region | quarter | units | revenue | margin |\n|--------|---------|-------|---------|--------|\n| EMEA   | Q1      | 1420  | 184600  | 0.312  |\n| EMEA   | Q2      | 1655  | 210430  | 0.298  |\n| AMER   | Q1      | 2310  | 298900  | 0.341  |\n| AMER   | Q2      | 2088  | 271440  | 0.336  |\n| APAC   | Q1      |  940  | 118200  | 0.276  |\n| APAC   | Q2      | 1102  | 141010  | 0.289  |"},
]


def load_prompts(path: str | Path | None = None) -> list[dict[str, str]]:
    """Load prompts from JSONL, or return the built-in set.

    Expected JSONL fields: text (required), domain (optional, defaults to
    "unknown"), id (optional, defaults to line order).
    """
    if path is None:
        return [
            {"id": str(i), "domain": p["domain"], "text": p["text"]}
            for i, p in enumerate(DEFAULT_PROMPTS)
        ]

    prompts: list[dict[str, str]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for i, line in enumerate(handle):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            prompts.append(
                {
                    "id": str(record.get("id", i)),
                    "domain": record.get("domain", "unknown"),
                    "text": record["text"],
                }
            )
    return prompts
