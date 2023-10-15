from typing import Any
import chex
import haiku as hk
import jax
import jax.numpy as jnp
import distrax
import optax
import rlax
import gymnax
from gymnax.environments.classic_control.cartpole import EnvParams, EnvState
from gymnax.environments.environment import Environment


@chex.dataclass(frozen=True)
class TimeStep:
    observation: chex.Array
    action: chex.Array
    discount: chex.Array
    reward: chex.Array
    behaviour_action_log_prob: chex.Array
    behaviour_value: chex.Array


def get_network_fn(num_outputs: int):
    """Returns a network function that can be used to compute the logits and value of the agent."""

    def network_fn(obs: chex.Array) -> chex.Array:
        x = hk.Sequential([hk.Linear(256), jax.nn.leaky_relu])(obs)

        logits = hk.Sequential(
            [hk.Linear(256), jax.nn.leaky_relu, hk.Linear(num_outputs)]
        )(x)

        value = hk.Sequential(
            [hk.Linear(256), jax.nn.leaky_relu, hk.Linear(1)]
        )(x)

        return logits, value

    return hk.without_apply_rng(hk.transform(network_fn))


def mutate_env_params(env_params: EnvParams, rng: chex.PRNGKey):
    """Mutates the parameters of the environment. Can be seen as a new Task. Very much a toy implementation."""
    keys = jax.random.split(rng, 8)
    noise_scale = 0.5
    return env_params.replace(
        gravity=9.8
        + jax.random.uniform(keys[0], minval=-noise_scale, maxval=noise_scale),
        masscart=1.0
        + jax.random.uniform(keys[1], minval=-noise_scale, maxval=noise_scale),
        masspole=0.1
        + jax.random.uniform(keys[2], minval=-noise_scale, maxval=noise_scale),
        total_mass=1.0
        + 0.1
        + jax.random.uniform(
            keys[3], minval=-noise_scale, maxval=noise_scale
        ),  # (masscart + masspole)
        length=0.5
        + jax.random.uniform(keys[4], minval=-noise_scale, maxval=noise_scale),
        polemass_length=0.05
        + jax.random.uniform(
            keys[5], minval=-noise_scale, maxval=noise_scale
        ),  # (masspole * length)
        force_mag=10.0
        + jax.random.uniform(keys[6], minval=-noise_scale, maxval=noise_scale),
        tau=0.02
        + jax.random.uniform(keys[7], minval=-noise_scale, maxval=noise_scale),
    )


def sample_tasks(base_env_params: EnvParams, rng_key, num_tasks):
    """Samples multiple tasks from the base environment params."""
    tasks_env_params = jax.vmap(mutate_env_params, in_axes=(None, 0))(
        base_env_params, jax.random.split(rng_key, num_tasks)
    )  # mutate env_params.
    return tasks_env_params


def get_meta_learner_fn(
    env: Environment,
    init_env_params: EnvParams,
    network_fn,
    opt_update_fn,
    rollout_len,
    agent_discount,
    iterations,
    gae_lambda,
    clip_epsilon,
    inner_loop_burn_in,
    num_trajectories,
    num_tasks,
    num_ppo_epochs=1,
):
    """Returns a meta-learner function that can be used to meta-train the parameters of the agent."""

    def rollout_fn(params, outer_rng, env_state, init_obs, env_params):
        """Performs an environment rollout."""

        def step_fn(carry, rng):
            """A single step of the environment."""

            obs_tm1, env_state, params, env_params = carry

            a_logits_tm1, v_tm1 = network_fn.apply(
                params, jnp.expand_dims(obs_tm1, 0)
            )
            a_tm1_dist = distrax.Categorical(a_logits_tm1[0])
            a_tm1 = a_tm1_dist.sample(seed=rng)
            a_tm1_log_prob = a_tm1_dist.log_prob(a_tm1)
            o_t, env_state, r_t, done, _ = env.step(
                rng, env_state, a_tm1, env_params
            )

            return (o_t, env_state, params, env_params), TimeStep(
                observation=obs_tm1,
                action=a_tm1,
                discount=1 - done,
                reward=r_t,
                behaviour_action_log_prob=a_tm1_log_prob,
                behaviour_value=v_tm1.squeeze(),
            )

        step_rngs = jax.random.split(outer_rng, rollout_len)
        (o_t, env_state, params, env_params), rollout = jax.lax.scan(
            step_fn, (init_obs, env_state, params, env_params), step_rngs
        )

        rollout = jax.tree_map(lambda x: jnp.expand_dims(x, 0), rollout)

        return rollout, env_state, o_t

    def sample_trajectory(params, rng_key, env_params):
        """Resets the environment and collects a single trajectory from the environment."""
        rng_key, rollout_rng = jax.random.split(rng_key)
        obs, env_state = env.reset(rng_key, env_params)
        rollout, _, _ = rollout_fn(
            params, rollout_rng, env_state, obs, env_params
        )
        return rollout

    def inner_policy_loss_fn(params: hk.Params, batch: TimeStep):
        """Computes the ppo-loss of the agent on a single trajectory."""
        batch = jax.tree_map(lambda x: jnp.squeeze(x, 0), batch)
        logits_t, v_t = network_fn.apply(params, batch.observation)
        a_log_prob_t = distrax.Categorical(logits_t).log_prob(
            batch.action.astype(jnp.int32)
        )
        adv_t = rlax.truncated_generalized_advantage_estimation(
            batch.reward[:-1],
            batch.discount[:-1] * agent_discount,
            gae_lambda,
            batch.behaviour_value,
            True,
        )
        rhos = jnp.exp(
            a_log_prob_t[:-1] - batch.behaviour_action_log_prob[:-1]
        )
        pg_loss = rlax.clipped_surrogate_pg_loss(rhos, adv_t, clip_epsilon)
        entropy_loss = distrax.Categorical(logits_t).entropy()[:-1]
        target_values = batch.behaviour_value[:-1] + adv_t
        value_loss = jnp.square(v_t[:-1] - target_values)
        loss = pg_loss + value_loss - 0.001 * entropy_loss
        loss = jnp.mean(loss)

        return loss

    def inner_policy_update_fn(
        inner_params: hk.Params,
        inner_opt_state: optax.OptState,
        rng_key: chex.PRNGKey,
        env_params: EnvParams,
    ):
        """Updates the parameters of the agent on a single task."""
        trajectory_keys = jax.random.split(rng_key, num_trajectories)
        rollout = jax.vmap(sample_trajectory, in_axes=(None, 0, None))(
            inner_params, trajectory_keys, env_params
        )

        def epoch(carry, _):
            inner_params, inner_opt_state, rollout = carry
            batch_loss = jax.vmap(inner_policy_loss_fn, in_axes=(None, 0))
            loss = lambda params, batch: jnp.mean(batch_loss(params, batch))
            (
                inner_loss,
                grads,
            ) = jax.value_and_grad(  # compute gradient on a single trajectory.
                loss
            )(
                inner_params, rollout
            )
            grads = jax.lax.pmean(
                grads, axis_name="tasks"
            )  # average gradients across tasks.
            grads = jax.lax.pmean(
                grads, axis_name="devices"
            )  # average gradients across devices.
            updates, new_opt_state = opt_update_fn.update(
                grads, inner_opt_state
            )  # transform grads.
            new_params = optax.apply_updates(
                inner_params, updates
            )  # update parameters.
            return (new_params, new_opt_state, rollout), inner_loss

        (new_params, new_opt_state, rollout), inner_loss = jax.lax.scan(
            epoch,
            (inner_params, inner_opt_state, rollout),
            None,
            length=num_ppo_epochs,
        )

        return new_params, new_opt_state, inner_loss

    def inner_loop_fn(
        meta_params: hk.Params,
        meta_opt_state: optax.OptState,
        rng_key: chex.PRNGKey,
        env_params,
    ):
        """Inner loop of maml - performs policy updates and returns the loss."""

        rng_key, burn_in_rng, update_rng = jax.random.split(rng_key, 3)

        def rollout_fn_step(carry, rng_key):
            inner_params, inner_opt_state, env_params = carry
            inner_params, inner_opt_state, _ = inner_policy_update_fn(
                inner_params, inner_opt_state, rng_key, env_params
            )
            return (inner_params, inner_opt_state, env_params), None

        (updated_inner_params, _, env_params), _ = jax.lax.scan(
            rollout_fn_step,
            (meta_params, meta_opt_state, env_params),
            jax.random.split(burn_in_rng, inner_loop_burn_in),
        )

        trajectory_keys = jax.random.split(update_rng, num_trajectories)
        rollout = jax.lax.stop_gradient(
            jax.vmap(sample_trajectory, in_axes=(None, 0, None))(
                updated_inner_params, trajectory_keys, env_params
            )
        )
        batch_loss = jax.vmap(inner_policy_loss_fn, in_axes=(None, 0))
        loss = jnp.mean(batch_loss(updated_inner_params, rollout))

        return loss

    def meta_loss(meta_params, meta_opt_state, inner_loop_key, task_env_param):
        """Computes the meta-loss."""
        outer_loss = inner_loop_fn(
            meta_params, meta_opt_state, inner_loop_key, task_env_param
        )
        outer_loss = jnp.mean(outer_loss)
        return outer_loss

    def outer_fn(
        meta_params: hk.Params,
        meta_opt_state: optax.OptState,
        rng_key: chex.PRNGKey,
        task_env_param: EnvParams,
    ):
        """Performs a single update of the meta-parameters."""
        rng_key, mutation_key, inner_loop_key = jax.random.split(rng_key, 3)

        # We get the loss and gradients of the meta objective.
        meta_loss_value, grads = jax.value_and_grad(meta_loss)(
            meta_params, meta_opt_state, inner_loop_key, task_env_param
        )
        grads = jax.lax.pmean(
            grads, axis_name="tasks"
        )  # average gradients across tasks.
        grads = jax.lax.pmean(
            grads, axis_name="devices"
        )  # average gradients across devices.
        updates, new_meta_opt_state = opt_update_fn.update(
            grads, meta_opt_state
        )  # transform grads.
        new_meta_params = optax.apply_updates(
            meta_params, updates
        )  # update parameters.

        return new_meta_params, new_meta_opt_state, meta_loss_value

    def meta_learner_fn(
        init_meta_params, init_meta_opt_state, rng_key: chex.PRNGKey
    ):
        """Performs multiple updates of the meta-agent."""

        # We sample a set of tasks.
        task_env_params = sample_tasks(init_env_params, rng_key, num_tasks)
        # We vmap the outer function optimisation loop across tasks.
        batched_outer_fn = jax.vmap(
            outer_fn, in_axes=(0, 0, 0, 0), axis_name="tasks"
        )

        def iterate_fn(
            carry, rng_key
        ):  # repeat many times to avoid going back to Python.
            meta_params, meta_opt_state, task_env_params = carry
            new_meta_params, new_meta_opt_state, meta_loss = batched_outer_fn(
                meta_params,
                meta_opt_state,
                jax.random.split(rng_key, num_tasks),
                task_env_params,
            )
            return (
                new_meta_params,
                new_meta_opt_state,
                task_env_params,
            ), meta_loss

        (
            new_meta_params,
            new_meta_opt_state,
            task_env_params,
        ), meta_loss = jax.lax.scan(
            iterate_fn,
            (init_meta_params, init_meta_opt_state, task_env_params),
            jax.random.split(rng_key, iterations),
        )
        return new_meta_params, new_meta_opt_state, jnp.mean(meta_loss)

    def fine_tune_fn(
        inner_params, inner_opt_state, rng_key, env_params, num_steps
    ):
        def rollout_fn_step(carry, rng_key):
            inner_params, inner_opt_state, env_params = carry
            inner_params, inner_opt_state, _ = inner_policy_update_fn(
                inner_params, inner_opt_state, rng_key, env_params
            )
            return (inner_params, inner_opt_state, env_params), None

        (updated_inner_params, _, env_params), _ = jax.lax.scan(
            rollout_fn_step,
            (inner_params, inner_opt_state, env_params),
            jax.random.split(rng_key, num_steps),
        )

        return updated_inner_params

    return meta_learner_fn, fine_tune_fn


def setup_experiment(seed, learning_rate):
    """Sets up the experiment."""

    key = jax.random.PRNGKey(seed)
    key, net_key = jax.random.split(key, 2)
    env, env_params = gymnax.make("CartPole-v1")

    dummy_obs = jnp.zeros(env.observation_space(env_params).shape)

    network_fn = get_network_fn(env.action_space(env_params).n)

    meta_params = network_fn.init(net_key, jnp.expand_dims(dummy_obs, 0))

    opt_update_fn = optax.adam(learning_rate)

    meta_opt_state = opt_update_fn.init(meta_params)

    return (
        meta_params,
        meta_opt_state,
        env,
        env_params,
        network_fn,
        opt_update_fn,
        key,
    )


def get_eval_fn(env: Environment, network_fn):
    """Returns a function that can be used to evaluate the agent on a single episode."""

    @jax.jit
    def eval_one_episode(params, rng, env_params):
        """Evaluates the agent on a single episode."""
        rng, reset_key = jax.random.split(rng)
        o_tm1, state = env.reset(reset_key, env_params)

        def step(val):
            params, state, o_tm1, tot_r, rng, _ = val
            rng, key_step = jax.random.split(rng)
            a_logits_t, _ = network_fn.apply(params, o_tm1[jnp.newaxis,])
            a_t = distrax.Categorical(a_logits_t).sample(seed=rng)[0]
            o_t, state, r_t, done, _ = env.step(
                key_step, state, a_t, env_params
            )
            tot_r += r_t
            return (params, state, o_t, tot_r, rng, done)

        params, state, o_t, tot_r, rng, done = jax.lax.while_loop(
            lambda val: val[5] == False,
            step,
            (params, state, o_tm1, 0, rng, False),
        )

        return params, tot_r

    return jax.vmap(eval_one_episode, in_axes=(None, 0, None))


def broadcast_to_shape(tree, shape):
    """Broadcasts x to the given shape."""
    return jax.tree_map(lambda x: jnp.broadcast_to(x, shape + x.shape), tree)


def extract_from_first_device(x):
    """Extracts the first element from the first device."""
    return jax.tree_map(lambda x: x[0, 0], x)


if __name__ == "__main__":
    META_TRAINING_STEPS = 10
    NUM_TASKS = 10
    NUM_EVAL_EPISODES = 200

    (
        meta_params,
        meta_opt_state,
        env,
        init_env_params,
        network_fn,
        opt_update_fn,
        key,
    ) = setup_experiment(seed=0, learning_rate=0.001)

    eval_fn = get_eval_fn(env, network_fn)

    key, eval_key = jax.random.split(key)

    eval_keys = jax.random.split(eval_key, 100)
    tot_reward = eval_fn(meta_params, eval_keys, init_env_params)[-1]

    print("Before Training Reward: ", jnp.mean(tot_reward))

    meta_learn_fn, fine_tune_fn = get_meta_learner_fn(
        env,
        init_env_params,
        network_fn,
        opt_update_fn,
        rollout_len=500,
        agent_discount=0.99,
        iterations=10,
        gae_lambda=0.95,
        clip_epsilon=0.3,
        num_trajectories=16,
        inner_loop_burn_in=2,
        num_tasks=NUM_TASKS,
    )

    meta_learn_fn = jax.pmap(meta_learn_fn, axis_name="devices")

    print("Meta-learning...")

    meta_params, meta_opt_state = broadcast_to_shape(
        (meta_params, meta_opt_state), (jax.local_device_count(), NUM_TASKS)
    )

    key, eval_key, fine_tune_key = jax.random.split(key, 3)

    for i in range(META_TRAINING_STEPS):
        key, train_key = jax.random.split(key)
        train_keys = jax.random.split(train_key, jax.local_device_count())
        reshape = lambda x: x.reshape(
            (jax.local_device_count(),) + x.shape[1:]
        )
        train_keys = reshape(jnp.stack(train_keys))

        meta_params, meta_opt_state, meta_loss = meta_learn_fn(
            meta_params, meta_opt_state, train_keys
        )
        print("Meta Loss: ", jnp.mean(meta_loss))

    fine_tune_keys = jax.random.split(
        fine_tune_key, jax.local_device_count() * NUM_TASKS
    )
    reshape = lambda x: x.reshape(
        (jax.local_device_count(), NUM_TASKS) + x.shape[1:]
    )
    fine_tune_keys = reshape(jnp.stack(fine_tune_keys))

    fine_tune_fn = jax.vmap(
        fine_tune_fn, axis_name="tasks", in_axes=(0, 0, 0, None, None)
    )
    fine_tune_fn = jax.pmap(
        fine_tune_fn,
        axis_name="devices",
        in_axes=(0, 0, 0, None, None),
        static_broadcasted_argnums=(4,),
    )
    fine_tuned_params = fine_tune_fn(
        meta_params, meta_opt_state, fine_tune_keys, init_env_params, 2
    )

    single_meta_params = extract_from_first_device(meta_params)
    single_fine_tuned_params = extract_from_first_device(fine_tuned_params)
    meta_opt_state = extract_from_first_device(meta_opt_state)

    eval_keys = jax.random.split(eval_key, NUM_EVAL_EPISODES)

    print("Evaluating...")

    tot_reward = eval_fn(single_meta_params, eval_keys, init_env_params)[-1]
    print("After Training - Base Meta Params: ", jnp.mean(tot_reward))

    tot_reward = eval_fn(single_fine_tuned_params, eval_keys, init_env_params)[
        -1
    ]
    print("After Training - Fine Tuned Meta Params: ", jnp.mean(tot_reward))
