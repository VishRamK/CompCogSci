# run_intuitive_gamer_connect4.py
import sys, json
from dataclasses import dataclass
import jax
import jax.numpy as jnp

from connect4_env import Connect4Environment
from igm_lite_connect4 import IGMLiteConnect4


@dataclass
class RolloutConfig:
    episodes: int = 10
    horizon: int = 64
    temperature: float = 0.35
    epsilon: float = 0.08


def _sample_action(rng, logits, legal_mask, temperature: float, epsilon: float):
    any_legal = jnp.any(legal_mask)

    def no_legal():
        return jnp.int32(-1), rng

    def yes_legal():
        rng1, rng2, rng3 = jax.random.split(rng, 3)
        do_rand = jax.random.uniform(rng1) < epsilon

        legal_count = jnp.sum(legal_mask)
        p_uniform = jnp.where(legal_mask, 1.0 / jnp.maximum(legal_count, 1), 0.0)
        rand_choice = jax.random.categorical(rng2, jnp.log(p_uniform)).astype(jnp.int32)

        very_neg = jnp.array(-1e9, dtype=logits.dtype)
        masked = jnp.where(legal_mask, logits, very_neg)
        t = jnp.maximum(jnp.array(temperature, dtype=logits.dtype), jnp.array(1e-6, dtype=logits.dtype))
        probs = jax.nn.softmax(masked / t)
        soft_choice = jax.random.categorical(rng3, jnp.log(probs)).astype(jnp.int32)

        action = jnp.where(do_rand, rand_choice, soft_choice)
        return action, rng3

    return jax.lax.cond(any_legal, yes_legal, no_legal)


def run_intuitive_gamer(connect4_json_path: str, num_episodes=10, json_log_path=None,
                       temperature=None, epsilon=None, horizon=None):
    cfg = RolloutConfig()
    cfg.episodes = num_episodes
    if temperature is not None: cfg.temperature = float(temperature)
    if epsilon is not None: cfg.epsilon = float(epsilon)
    if horizon is not None: cfg.horizon = int(horizon)

    env = Connect4Environment(connect4_json_path)
    igm = IGMLiteConnect4(env)

    summary = {
        "config": {
            "episodes": cfg.episodes,
            "horizon": cfg.horizon,
            "temperature": cfg.temperature,
            "epsilon": cfg.epsilon,
            "game_path": connect4_json_path,
            "rows": env.R,
            "cols": env.C,
        },
        "episodes": [],
        "counts": {"P0": 0, "P1": 0, "draw": 0, "timeout": 0, "no_legal_moves": 0, "unknown_winner": 0},
        "lengths": [],
    }

    for ep_i in range(cfg.episodes):
        rng = jax.random.PRNGKey(ep_i)
        state = env.init(rng)
        moves = []
        outcome = None
        steps = 0

        while not bool(jax.device_get(state.terminated)):
            if steps >= cfg.horizon:
                outcome = "timeout"
                summary["counts"]["timeout"] += 1
                break

            logits = igm.policy(state)
            legal = state.legal_action_mask.astype(bool)

            action, rng = _sample_action(rng, logits, legal, cfg.temperature, cfg.epsilon)
            if int(action) < 0:
                outcome = "no_legal_moves"
                summary["counts"]["no_legal_moves"] += 1
                break

            moves.append(int(action))
            state = env.step(state, jnp.asarray(action, dtype=jnp.int32))
            steps += 1

        value_est = float(igm.value(state))

        if outcome is None:
            if int(state.winner) == 0:
                outcome = "P0"; summary["counts"]["P0"] += 1
            elif int(state.winner) == 1:
                outcome = "P1"; summary["counts"]["P1"] += 1
            elif int(state.winner) == -1 and bool(state.terminated):
                outcome = "draw"; summary["counts"]["draw"] += 1
            else:
                outcome = "unknown_winner"; summary["counts"]["unknown_winner"] += 1

        ep = {
            "episode": ep_i + 1,
            "outcome": outcome,
            "winner_field": int(state.winner),
            "value_est": value_est,
            "moves": moves,
            "steps": len(moves),
        }
        summary["episodes"].append(ep)
        summary["lengths"].append(len(moves))

        print(f"Episode {ep['episode']}: {outcome}")
        print(f"  Winner field: {ep['winner_field']}")
        print(f"  Value estimate: {ep['value_est']:.3f}")
        print(f"  Steps: {ep['steps']}  |  Moves (cols): {moves}")
        print("-" * 60)

    print("Done.")
    if json_log_path:
        with open(json_log_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"Saved gameplay log to {json_log_path}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python run_intuitive_gamer_connect4.py path/to/connect4.json [--episodes N] [--json path] [--temp T] [--eps E] [--horizon H]")
        sys.exit(1)

    path = sys.argv[1]
    num_episodes = 10
    json_log_path = None
    temperature = None
    epsilon = None
    horizon = None

    args = sys.argv[2:]
    i = 0
    while i < len(args):
        if args[i] == "--episodes" and i + 1 < len(args):
            num_episodes = int(args[i+1]); i += 2
        elif args[i] == "--json" and i + 1 < len(args):
            json_log_path = args[i+1]; i += 2
        elif args[i] == "--temp" and i + 1 < len(args):
            temperature = float(args[i+1]); i += 2
        elif args[i] == "--eps" and i + 1 < len(args):
            epsilon = float(args[i+1]); i += 2
        elif args[i] == "--horizon" and i + 1 < len(args):
            horizon = int(args[i+1]); i += 2
        else:
            i += 1

    run_intuitive_gamer(path, num_episodes, json_log_path, temperature, epsilon, horizon)


if __name__ == "__main__":
    main()
