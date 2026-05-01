Data Generation
---------------

This workflow covers annotating and generating the demonstration dataset using
`Isaac Lab Mimic <https://isaac-sim.github.io/IsaacLab/main/source/overview/imitation-learning/teleop_imitation.html>`_.


**Docker Container**: Base (see :doc:`../../quickstart/installation` for more details)

:docker_run_default:


Step 1: Annotate Demonstrations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This step describes how to annotate the demonstrations recorded in the preceding step
so they can be used by Isaac Lab Mimic. For more details on Mimic annotation, see the
`Isaac Lab Mimic documentation <https://isaac-sim.github.io/IsaacLab/main/source/overview/imitation-learning/teleop_imitation.html#annotate-the-demonstrations>`_.

To start the annotation process, run the following command:

.. code-block:: bash

   python isaaclab_arena/scripts/imitation_learning/annotate_demos.py \
     --viz kit \
     --device cpu \
     --input_file $DATASET_DIR/arena_g1_static_apple_dataset_recorded.hdf5 \
     --output_file $DATASET_DIR/arena_g1_static_apple_dataset_annotated.hdf5 \
     --mimic \
     galileo_g1_static_pick_and_place \
     --object apple_01_objaverse_robolab \
     --destination clay_plates_hot3d_robolab

Follow the instructions described on the CLI to complete the annotation.

.. note::

   The static Mimic config (``StaticPickAndPlaceMimicEnvCfg``) inherits the per-arm subtask
   sequence from the loco-manip variant and only overrides the body subtask group: the loco-manip's
   four navigation phases (``navigate_to_table -> navigate_turn_inplace -> navigate_to_bin -> final``)
   are collapsed into a single no-op subtask spanning the whole demo. This is required because the
   nav termination signals never fire in the static env (the robot never moves its base), so a
   four-phase body group would deadlock Mimic at annotation time.


Step 2: Generate Augmented Dataset
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Isaac Lab Mimic generates additional demonstrations from the annotated demonstrations
by applying object and trajectory transformations to introduce data variations.

Generate the dataset:

.. code-block:: bash

   # Generate 100 demonstrations
   python isaaclab_arena/scripts/imitation_learning/generate_dataset.py \
     --headless \
     --enable_cameras \
     --mimic \
     --input_file $DATASET_DIR/arena_g1_static_apple_dataset_annotated.hdf5 \
     --output_file $DATASET_DIR/arena_g1_static_apple_dataset_generated.hdf5 \
     --generation_num_trials 100 \
     --device cpu \
     galileo_g1_static_pick_and_place \
     --object apple_01_objaverse_robolab \
     --destination clay_plates_hot3d_robolab \
     --embodiment g1_wbc_pink

Data generation takes 1-4 hours depending on your CPU/GPU.
You can remove ``--headless`` and add ``--viz kit``
(before specifying the task name ``galileo_g1_static_pick_and_place``) to visualize during data generation.

.. note::

   The static env writes its dataset under a distinct ``static_pick_and_place_*`` datagen-name prefix
   (configured in ``StaticPickAndPlaceMimicEnvCfg.__post_init__``), so it cannot accidentally
   overwrite a loco-manipulation dataset that uses the ``locomanip_pick_and_place_*`` prefix —
   even if both runs share the same ``$DATASET_DIR``. The recorder patch
   (``patch_recorders()`` from ``isaaclab_arena/utils/locomanip_mimic_patch.py``) is registered
   automatically by ``GalileoG1StaticPickAndPlaceEnvironment.get_env()`` when ``--mimic`` is passed,
   so generated HDF5 files contain the ``"action"`` key the converter / training pipeline expects.


Step 3: Validate Generated Dataset (Optional)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To visualize the data produced, you can replay the dataset using the following command:

.. code-block:: bash

   python isaaclab_arena/scripts/imitation_learning/replay_demos.py \
     --viz kit \
     --device cpu \
     --enable_cameras \
     --dataset_file $DATASET_DIR/arena_g1_static_apple_dataset_generated.hdf5 \
     galileo_g1_static_pick_and_place \
     --object apple_01_objaverse_robolab \
     --destination clay_plates_hot3d_robolab \
     --embodiment g1_wbc_pink

You should see the robot successfully perform the task.

.. note::

   The dataset was generated using CPU device physics, therefore the replay uses ``--device cpu`` to ensure reproducibility.
