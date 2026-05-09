Closed-Loop Policy Inference and Evaluation
-------------------------------------------

This workflow demonstrates running the finetuned GR00T N1.7 policy in closed-loop and evaluating it
in the Arena Unitree G1 Static Apple-to-Plate Task environment using Arena's **server-client (remote-policy)
architecture**. The server hosts the finetuned checkpoint outside the Arena container; the Arena
container runs the simulation and queries the server over ZeroMQ.

Note that this tutorial assumes that you've completed the
:doc:`preceding step (Policy Training) <step_3_policy_training>`.


Step 0: Start the GR00T policy server
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The server runs Arena's ``Gr00tRemoteServerSidePolicy`` (which wraps GR00T's ``Gr00tPolicy``) on top
of the standalone Isaac-GR00T (N1.7) Python package. Start it **before** launching the client; the
client will connect on first inference.

The server is configured by a YAML at
``isaaclab_arena_gr00t/policy/config/g1_static_apple_gr00t_closedloop_config.yaml``.

.. dropdown:: Server-side configuration file (``g1_static_apple_gr00t_closedloop_config.yaml``)
   :animate: fade-in

   .. code-block:: yaml

      # Path on the server's filesystem (or container mount) to the finetuned checkpoint dir.
      model_path: /models/isaaclab_arena/static_apple_tutorial/static_apple_n17_finetune/checkpoint-20000
      language_instruction: "Pick up the apple from the shelf and place it onto the plate on the same shelf next to it."

      # Must match the diffusion head's action_horizon baked into the finetuned checkpoint.
      action_horizon: 40

      # The N1.7 finetune from step 3 uses the NEW_EMBODIMENT tag.
      embodiment_tag: NEW_EMBODIMENT

      video_backend: decord
      modality_config_path: isaaclab_arena_gr00t/embodiments/g1/g1_sim_wbc_data_gr00t_n_1_7_config.py

      policy_joints_config_path: isaaclab_arena_gr00t/embodiments/g1/gr00t_43dof_joint_space.yaml
      action_joints_config_path: isaaclab_arena_gr00t/embodiments/g1/43dof_joint_space.yaml
      state_joints_config_path: isaaclab_arena_gr00t/embodiments/g1/43dof_joint_space.yaml

      # Number of actions to execute before next inference; <= action_horizon.
      action_chunk_length: 40
      pov_cam_name_sim: "robot_head_cam_rgb"

      task_mode_name: g1_locomanipulation


Run the server **outside Docker** in the standalone Isaac-GR00T (N1.7) venv created in
:doc:`index`. This is the simplest setup once the venv exists: no container build, no
``GROOT_DEPS_DIR`` overrides, just the standalone repo's own dependencies.

From the host, with ``$ISAAC_GR00T_DIR`` and Arena both checked out:

The standalone Isaac-GR00T repo provides the ``gr00t`` package; Arena provides
``gr00t_remote_policy`` and the ZeroMQ server entrypoint. Add Arena to ``PYTHONPATH`` so the
standalone venv can import Arena's server modules without installing Arena into the venv.
Then launch Arena's server with the static-apple YAML from inside the standalone checkout's
``uv``-managed environment.

.. code-block:: bash

   export ISAAC_GR00T_DIR=/path/to/Isaac-GR00T
   cd /path/to/IsaacLab-Arena
   export PYTHONPATH=$PWD:${PYTHONPATH:-}

   uv run --project $ISAAC_GR00T_DIR python -m isaaclab_arena.remote_policy.remote_policy_server_runner \
      --policy_type isaaclab_arena_gr00t.policy.gr00t_remote_policy.Gr00tRemoteServerSidePolicy \
      --policy_config_yaml_path /path/to/IsaacLab-Arena/isaaclab_arena_gr00t/policy/config/g1_static_apple_gr00t_closedloop_config.yaml \
      --host 0.0.0.0 \
      --port 5555

The server prints ``[Gr00tRemoteServerSidePolicy] config:`` followed by the parsed YAML and
then ``listening on 0.0.0.0:5555`` once it is ready for clients.


Step 1: Run Single Environment Evaluation (Arena container)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

With the server from Step 0 running, launch the Arena client. The client side does not need any
GR00T dependencies — it talks to the server over ZeroMQ — so it runs in the standard **Base**
Arena container.

**Docker Container**: Base (see :doc:`../../quickstart/installation` for more details)

:docker_run_default:

Once inside the container, set the dataset and models directories.

.. code:: bash

    export DATASET_DIR=/datasets/isaaclab_arena/static_apple_tutorial
    export MODELS_DIR=/models/isaaclab_arena/static_apple_tutorial

We first run the policy in a single environment with visualization via the GUI. Replace
``<SERVER_HOST>`` below with the IP of the host running Step 0 (or ``localhost`` if it is the same
machine).

.. code-block:: bash

   python isaaclab_arena/evaluation/policy_runner.py \
     --viz kit \
     --policy_type isaaclab_arena.policy.action_chunking_client.ActionChunkingClientSidePolicy \
     --remote_host <SERVER_HOST> \
     --remote_port 5555 \
     --num_steps 600 \
     --device cpu \
     --enable_cameras \
     galileo_g1_static_pick_and_place \
     --object apple_01_objaverse_robolab \
     --destination clay_plates_hot3d_robolab \
     --embodiment g1_wbc_agile_joint

Note the lower ``--num_steps`` (600 instead of 1500): with no walking phase, a successful
static apple-to-plate episode runs for roughly half as long as the loco-manipulation variant.

Note also that the client command does **not** take a ``--policy_config_yaml_path``: the YAML is
the server's concern, and the client only needs to know where the server is listening. The
``ActionChunkingClientSidePolicy`` does the action-chunking buffering on the client side; it expects
the server to emit fixed-length chunks of ``action_horizon`` actions per inference (40 here), which
the YAML configures.

The evaluation should produce the following output on the console at the end of the evaluation.
You should see similar metrics.

Note that all these metrics are computed over the entire evaluation process, and are affected
by the quality of post-trained policy, the quality of the dataset, and number of steps in the evaluation.

.. code-block:: text

   [Rank 0/1] Metrics: {'success_rate': 1.0, 'num_episodes': 1}

Run Parallel Environments Evaluation (Optional)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Parallel evaluation of the policy in multiple parallel environments is also supported by the policy
runner. Both tabs below assume the server from Step 0 is still running.

.. tab-set::

   .. tab-item:: Single GPU Evaluation

      Test the policy in 5 parallel environments with visualization via the GUI run:

      .. code-block:: bash

         python isaaclab_arena/evaluation/policy_runner.py \
           --viz kit \
           --policy_type isaaclab_arena.policy.action_chunking_client.ActionChunkingClientSidePolicy \
           --remote_host <SERVER_HOST> \
           --remote_port 5555 \
           --num_steps 500 \
           --num_envs 5 \
           --enable_cameras \
           --device cuda \
           galileo_g1_static_pick_and_place \
           --object apple_01_objaverse_robolab \
           --destination clay_plates_hot3d_robolab \
           --embodiment g1_wbc_agile_joint

   .. tab-item:: Distribute Multi-GPU Evaluation

      Test the policy in 5 parallel environments on each GPU with 2 GPUs total run:

      .. code-block:: bash

         python -m torch.distributed.run --nnodes=1 --nproc_per_node=2 isaaclab_arena/evaluation/policy_runner.py \
           --policy_type isaaclab_arena.policy.action_chunking_client.ActionChunkingClientSidePolicy \
           --remote_host <SERVER_HOST> \
           --remote_port 5555 \
           --num_steps 500 \
           --num_envs 5 \
           --enable_cameras \
           --device cuda \
           --distributed \
           --headless \
           galileo_g1_static_pick_and_place \
           --object apple_01_objaverse_robolab \
           --destination clay_plates_hot3d_robolab \
           --embodiment g1_wbc_agile_joint

.. note::

   With the server-client architecture, ``--policy_device`` is no longer a client-side concern: the
   server places the policy on its own GPU (``policy_device`` in the server YAML, default
   ``cuda``). The client's ``--device`` flag still controls Arena's physics backend.

And during the evaluation, you should see the following output on the console at the end of the evaluation
indicating which environments are terminated (task-specific conditions like the apple is placed onto the plate,
or the episode length is exceeded by 30 seconds),
or truncated (if timeouts are enabled, like the maximum episode length is exceeded).

.. code-block:: text

   Resetting policy for terminated env_ids: tensor([3], device='cuda:0') and truncated env_ids: tensor([], device='cuda:0', dtype=torch.int64)

At the end of the evaluation, you should see the following output on the console indicating the metrics.
You can see that the success rate might not be 1.0 as more trials are being evaluated and randomizations are being introduced,
and the number of episodes is more than the single environment evaluation because of the parallelization.

.. code-block:: text

   [Rank 0/1] Metrics: {'success_rate': 1.0, 'num_episodes': 4}

.. note::

   Note that the embodiment used in closed-loop policy inference is ``g1_wbc_agile_joint``, which is
   different from ``g1_wbc_agile_pink`` used during teleoperation recording.
   This is because during tele-operation, the upper body is controlled via target end-effector poses,
   which are realized by using the PINK IK controller, and the lower body is controlled via the AGILE
   WBC policy. The GR00T N1.7 policy is trained on upper body joint positions and lower body WBC
   policy inputs, so we use the joint-control twin (``g1_wbc_agile_joint``) for closed-loop policy
   inference -- it shares the AGILE lower-body backend with the recording embodiment, just bypasses
   PinkIK.

.. note::

   The example policy was trained on datasets recorded with CPU-based physics, so the
   single-environment command above uses ``--device cpu`` to keep evaluation physics aligned
   with training and give per-episode reproducibility. The parallel commands instead use
   ``--device cuda`` for throughput -- this swaps the physics backend, so individual episodes
   are no longer bit-for-bit reproducible against the CPU-trained policy, but aggregate
   success-rate metrics over many episodes remain informative. If your dataset was recorded on
   GPU physics, prefer ``--device cuda`` for both single and parallel runs to keep evaluation
   physics aligned with training.

.. note::

   The same-shelf placement makes the static variant slightly easier than the loco-manipulation
   apple-to-plate task: the destination plate is always within arm's reach so the policy
   never has to recover from a mistimed approach, and there are no intermediate locomotion
   phases that can drift off-course. The success criterion is the same contact-sensor
   termination used by the loco-manipulation variant (``force_threshold=0.5 N``,
   ``velocity_threshold=0.1 m/s``), filtered to contacts with the ``--destination`` asset.
   Both values are passed to ``PickAndPlaceTask`` from
   ``isaaclab_arena_environments/galileo_g1_static_pick_and_place_environment.py``; edit the
   ``force_threshold`` / ``velocity_threshold`` kwargs there if you need a different success
   criterion for a new pick-up object or destination.

.. note::

   **Common server-client failure modes.**

   - ``ValueError: Invalid action shape, expected: 23, received: 50.`` — the client's embodiment
     expects a 23-D PinkIK action, but the server is returning a 43-DoF joint chunk. Make sure the
     client uses ``--embodiment g1_wbc_agile_joint`` (joint twin), not
     ``g1_wbc_agile_pink`` (PinkIK twin).
   - ``ModuleNotFoundError: No module named '...gr00t_remote_closedloop_policy'`` on the client
     side — the client's ``--policy_type`` is wrong. The remote-policy *client* is
     ``isaaclab_arena.policy.action_chunking_client.ActionChunkingClientSidePolicy``;
     ``Gr00tRemoteServerSidePolicy`` is the **server-side** class.
   - Action shape mismatch on the server (``Action key 'left_arm''s horizon must be 50. Got 40``)
     — the action modality registered at training time disagrees with the modality registered at
     server boot. Re-finetune at the same horizon or update the modality config to match the
     checkpoint (see the caution in :doc:`step_3_policy_training`).
