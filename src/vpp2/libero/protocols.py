"""Publication protocols. No simulator imports or task-success redefinitions."""

BASE_SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
PROTOCOLS = {
    "standard": dict(
        name="original2000",
        trials=50,
        seed=42,
        settle=30,
        reset="task_init_state",
        seed_mode="sequential",
        init_start=0,
        episodes=2000,
    ),
    "ood": dict(
        name="paper4500",
        trials=50,
        seed=7,
        settle=10,
        reset="random",
        seed_mode="task_episode_hash",
        init_start=0,
        episodes=4500,
    ),
    "pro": dict(
        name="public800",
        trials=10,
        seed=42,
        settle=15,
        reset="task_init_state",
        seed_mode="benchmark_seed_sequence",
        init_start=1,
        episodes=800,
    ),
}
REVISIONS = {
    "standard": "8f1084e3132a39270c3a13ebe37270a43ece2a01",
    "ood": "587a6cbf64f16c7b87fa5805dc0ed934192239a4",
    "pro": "c86fc3b8293185a6f373677018ff3e37f8391602",
}


def cells(benchmark):
    if benchmark == "standard":
        return [(None, suite, task) for suite in BASE_SUITES for task in range(10)]
    if benchmark == "ood":
        return [
            (seed, suite + "_ood", task)
            for seed in range(3)
            for suite in BASE_SUITES[:3]
            for task in range(10)
        ]
    if benchmark == "pro":
        return [
            (None, suite + "_" + axis, task)
            for suite in BASE_SUITES
            for axis in ("swap", "task")
            for task in range(10)
        ]
    raise ValueError(f"Unknown benchmark: {benchmark}")


def validate_config(cfg):
    if cfg.benchmark not in PROTOCOLS or cfg.protocol != PROTOCOLS[cfg.benchmark]["name"]:
        raise ValueError("Benchmark/protocol mismatch")
    sampler = cfg.get("action_sampler", "flow_matching")
    if sampler not in {"flow_matching", "meanflow", "direct_action"}:
        raise ValueError("Unknown action sampler")
    if sampler in {"meanflow", "direct_action"}:
        if int(cfg.get("action_nfe", 0)) != 1 or int(cfg.get("source_action_step", -1)) not in {0, 30000}:
            raise ValueError("One-step evaluation requires action_nfe=1 and source_action_step=0 (fresh) or 30000 (continuation)")
        if int(cfg.action_step) <= 0:
            raise ValueError("action_step must be the actual positive student update count")
    elif int(cfg.get("action_nfe", 10)) not in {1, 10}:
        raise ValueError("FM baseline action_nfe must be 10 (B0) or 1 (B1)")
    elif cfg.get("fm_continuation", False):
        if int(cfg.get("source_action_step", 0)) != 30000 or int(cfg.action_step) <= 0:
            raise ValueError(
                "B2 requires source_action_step=30000 and its actual positive update count"
            )
    elif int(cfg.action_step) != 30000:
        raise ValueError("FM release requires Action 30k")
    if int(cfg.video_step) != 10000:
        raise ValueError("LIBERO uses the paired frozen Video 10k")
    return PROTOCOLS[cfg.benchmark]
