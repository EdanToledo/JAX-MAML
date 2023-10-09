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
    
    def network_fn(obs: chex.Array) -> chex.Array:
        x = hk.Sequential([
            hk.Linear(128),
            jax.nn.relu])(obs)

        logits = hk.Sequential([
            hk.Linear(128), 
            jax.nn.relu,
            hk.Linear(num_outputs)])(x)
        
        value = hk.Sequential([
            hk.Linear(128), 
            jax.nn.relu,
            hk.Linear(1)])(x)

        return logits, value
    
    return hk.without_apply_rng(hk.transform(network_fn))

def mutate_env_params(env_params : EnvParams, rng : chex.PRNGKey):
    """Mutates the parameters of the environment. Can be seen as a new Task."""
    keys = jax.random.split(rng, 8)
    noise_scale = 0.0
    return env_params.replace(  gravity = 9.8 + jax.random.uniform(keys[0], minval=-noise_scale, maxval=noise_scale),
                                masscart = 1.0 + jax.random.uniform(keys[0], minval=-noise_scale, maxval=noise_scale),
                                masspole = 0.1 + jax.random.uniform(keys[0], minval=-noise_scale, maxval=noise_scale),
                                total_mass = 1.0 + 0.1 + jax.random.uniform(keys[0], minval=-noise_scale, maxval=noise_scale),  # (masscart + masspole)
                                length = 0.5 + jax.random.uniform(keys[0], minval=-noise_scale, maxval=noise_scale),
                                polemass_length = 0.05 + jax.random.uniform(keys[0], minval=-noise_scale, maxval=noise_scale),  # (masspole * length)
                                force_mag = 10.0 + jax.random.uniform(keys[0], minval=-noise_scale, maxval=noise_scale),
                                tau = 0.02 + jax.random.uniform(keys[0], minval=-noise_scale, maxval=noise_scale))

def sample_tasks(base_env_params : EnvParams, rng_key, num_tasks):
    """Samples multiple tasks from the base environment params."""
    tasks_env_params = jax.vmap(mutate_env_params, in_axes=(None, 0))(base_env_params, jax.random.split(rng_key, num_tasks))  # mutate env_params.
    return tasks_env_params


def get_meta_learner_fn(
    env : Environment, init_env_params : EnvParams, network_fn, opt_update_fn, rollout_len, agent_discount, iterations, gae_lambda, clip_epsilon, inner_loop_burn_in, num_trajectories, num_tasks):
    """Returns a meta-learner function that can be used to meta-train the parameters of the agent."""

    def rollout_fn(params, outer_rng, env_state, init_obs, env_params):
        """Performs an environment rollout."""

        def step_fn(carry, rng):
            """A single step of the environment."""
            
            obs_tm1, env_state, params, env_params = carry
        
            a_logits_tm1, v_tm1 = network_fn.apply(params, jnp.expand_dims(obs_tm1, 0))  
            a_tm1_dist = distrax.Categorical(a_logits_tm1[0])
            a_tm1 = a_tm1_dist.sample(seed=rng) 
            a_tm1_log_prob = a_tm1_dist.log_prob(a_tm1)  
            o_t, env_state, r_t, done, _ = env.step(rng, env_state, a_tm1, env_params)
            
            return (o_t, env_state, params, env_params), TimeStep( 
            observation = obs_tm1, action=a_tm1, discount=1-done, reward=r_t, behaviour_action_log_prob=a_tm1_log_prob, behaviour_value=v_tm1.squeeze())
            

        step_rngs = jax.random.split(outer_rng, rollout_len)
        (o_t, env_state, params, env_params), rollout = jax.lax.scan(step_fn, (init_obs, env_state, params, env_params), step_rngs)  

        rollout = jax.tree_map(lambda x: jnp.expand_dims(x,0), rollout)
        
        return rollout, env_state, o_t
    
    def sample_trajectory(params, rng_key, env_params):
        """Resets the environment and collects a single trajectory from the environment."""
        rng_key, rollout_rng = jax.random.split(rng_key)
        obs, env_state = env.reset(rng_key, env_params)
        rollout, _, _ = rollout_fn(params, rollout_rng, env_state, obs, env_params)
        return rollout
        
    def inner_policy_loss_fn(params : hk.Params, batch : TimeStep):
        """Computes the loss of the agent on a single trajectory."""
        batch = jax.tree_map(lambda x: jnp.squeeze(x, 0), batch)
        logits_t, v_t = network_fn.apply(params, batch.observation)  
        a_log_prob_t = distrax.Categorical(logits_t).log_prob(batch.action.astype(jnp.int32)) 
        adv_t = rlax.truncated_generalized_advantage_estimation(batch.reward[:-1], batch.discount[1:] * agent_discount, gae_lambda, batch.behaviour_value, True) 
        rhos = jnp.exp(a_log_prob_t[:-1] - batch.behaviour_action_log_prob[:-1])
        pg_loss = rlax.clipped_surrogate_pg_loss(rhos, adv_t, clip_epsilon)
        entropy_loss = distrax.Categorical(logits_t).entropy()[:-1]
        target_values = batch.behaviour_value[:-1] + adv_t  
        value_loss = jnp.square(v_t[:-1] - target_values)
        loss = pg_loss + value_loss - 0.001*entropy_loss 
        loss = jnp.mean(loss)
    
        return loss
    
    def inner_policy_update_fn(inner_params : hk.Params, inner_opt_state : optax.OptState, rng_key : chex.PRNGKey, env_params : EnvParams):
        """Updates the parameters of the agent on a single task."""
        trajectory_keys = jax.random.split(rng_key, num_trajectories)
        rollout = jax.vmap(sample_trajectory, in_axes=(None, 0, None))(inner_params, trajectory_keys, env_params)
        batch_loss = jax.vmap(inner_policy_loss_fn, in_axes=(None, 0))
        loss = lambda params, batch: jnp.mean(batch_loss(params, batch))
        grads = jax.grad(  # compute gradient on a single trajectory.
            loss)(inner_params, rollout)
        updates, new_opt_state = opt_update_fn.update(grads, inner_opt_state)  # transform grads.
        new_params = optax.apply_updates(inner_params, updates)  # update parameters.
       
        return new_params, new_opt_state


    def inner_loop_fn(meta_params : hk.Params, meta_opt_state : optax.OptState, rng_key : chex.PRNGKey, env_params):
        """Inner loop of maml - performs policy updates and returns the loss."""
        
        rng_key, burn_in_rng, update_rng = jax.random.split(rng_key, 3)

        def rollout_fn_step(carry, rng_key):
            inner_params, inner_opt_state, env_params = carry
            inner_params, inner_opt_state = inner_policy_update_fn(inner_params, inner_opt_state, rng_key, env_params)
            return (inner_params, inner_opt_state, env_params), None

        (updated_inner_params, _, env_params), _ = jax.lax.scan(rollout_fn_step, (meta_params, meta_opt_state, env_params), jax.random.split(burn_in_rng, inner_loop_burn_in))
            
        trajectory_keys = jax.random.split(update_rng, num_trajectories)
        rollout = jax.vmap(sample_trajectory, in_axes=(None, 0, None))(updated_inner_params, trajectory_keys, env_params)
        batch_loss = jax.vmap(inner_policy_loss_fn, in_axes=(None, 0))
        loss = jnp.mean(batch_loss(updated_inner_params, rollout))

        return loss
    
    def meta_loss(meta_params, meta_train_state, inner_loop_key, task_env_params):
            """Computes the meta-loss."""
            outer_loss = jax.vmap(inner_loop_fn, in_axes=(None, None, 0, 0))(meta_params, meta_train_state, jax.random.split(inner_loop_key, num_tasks), task_env_params)
            outer_loss = jnp.mean(outer_loss)
            return outer_loss
        
    def outer_fn(meta_params : hk.Params, meta_opt_state : optax.OptState, rng_key : chex.PRNGKey):
        """Performs a single update of the meta-parameters."""
        rng_key, mutation_key, inner_loop_key = jax.random.split(rng_key, 3)
        task_env_params = sample_tasks(init_env_params, mutation_key, num_tasks)
        
        meta_loss_value, grads = jax.value_and_grad(meta_loss)(meta_params, meta_opt_state, inner_loop_key, task_env_params)
       
        updates, new_meta_opt_state = opt_update_fn.update(grads, meta_opt_state)  # transform grads.
        new_meta_params = optax.apply_updates(meta_params, updates)  # update parameters.

        return new_meta_params, new_meta_opt_state, meta_loss_value
        
    def meta_learner_fn(init_meta_params, init_meta_opt_state, rng_key : chex.PRNGKey):
        """Performs multiple updates of the meta-agent."""

        def iterate_fn(meta_train_state, rng_key):  # repeat many times to avoid going back to Python.
            meta_params, meta_opt_state = meta_train_state
            new_meta_params, new_meta_opt_state, meta_loss = outer_fn(meta_params, meta_opt_state, rng_key)
            return (new_meta_params, new_meta_opt_state), meta_loss
        
        (new_meta_params, new_meta_opt_state), meta_loss = jax.lax.scan(iterate_fn, (init_meta_params, init_meta_opt_state), jax.random.split(rng_key, iterations))
        return new_meta_params, new_meta_opt_state, jnp.mean(meta_loss)

    return meta_learner_fn

def setup_experiment(seed, learning_rate):
    """Sets up the experiment."""

    key = jax.random.PRNGKey(seed)
    key, net_key = jax.random.split(key, 2)
    env, env_params = gymnax.make("CartPole-v1")
    
    dummy_obs = jnp.zeros(env.observation_space(env_params).shape)

    network_fn = get_network_fn(env.action_space(env_params).n)

    meta_params = network_fn.init(net_key, jnp.expand_dims(dummy_obs, 0))

    opt_update_fn = optax.sgd(learning_rate, momentum=0.9)

    meta_opt_state = opt_update_fn.init(meta_params)

    return meta_params, meta_opt_state, env, env_params, network_fn, opt_update_fn, key

meta_params, meta_opt_state, env, init_env_params, network_fn, opt_update_fn, key = setup_experiment(seed=0, learning_rate=0.001)


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
        o_t, state, r_t, done, _  = env.step(key_step, state, a_t, env_params)
        tot_r += r_t
        return (params, state, o_t, tot_r, rng, done)
       
    params, state, o_t, tot_r, rng, done = jax.lax.while_loop(lambda val : val[5] == False, step, (params, state, o_tm1, 0, rng, False))

    return params, tot_r


key, eval_key = jax.random.split(key)

eval_keys = jax.random.split(eval_key, 100)
tot_reward = jax.vmap(eval_one_episode, in_axes=(None, 0, None))(meta_params, eval_keys, init_env_params)[-1]

print("Before Training Reward: ", jnp.mean(tot_reward))

meta_learn_fn = get_meta_learner_fn(env,
                                    init_env_params, 
                                    network_fn, 
                                    opt_update_fn, 
                                    rollout_len=64, 
                                    agent_discount=0.99, 
                                    iterations=100, 
                                    gae_lambda=0.95, 
                                    clip_epsilon=0.2,
                                    num_trajectories=64,
                                    inner_loop_burn_in=1,
                                    num_tasks=1)

meta_learn_fn = jax.jit(meta_learn_fn)

meta_training_steps = 5
print("Meta-learning...")

for i in range(meta_training_steps):
    meta_params, meta_opt_state, meta_loss = meta_learn_fn(meta_params, meta_opt_state, key)
    print("Meta Loss: ", meta_loss)

key, eval_key = jax.random.split(key)
eval_keys = jax.random.split(eval_key, 100)
tot_reward = jax.vmap(eval_one_episode, in_axes=(None, 0, None))(meta_params, eval_keys, init_env_params)[-1]

print("After Training Reward: ", jnp.mean(tot_reward))
