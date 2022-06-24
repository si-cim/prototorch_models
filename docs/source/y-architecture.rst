.. Documentation of the updated Architecture.

Proto Y Architecture
========================================

Overview
****************************************

The Proto Y Architecture is a framework for abstract prototype learning methods.

It divides the problem into multiple steps:

    * **Components** : Recalling the position and metadata of the components/prototypes.
    * **Backbone** : Apply a mapping function to data and prototypes.
    * **Comparison** : Calculate a dissimilarity based on the latent positions.
    * **Competition** : Calculate competition values based on the comparison and the metadata.
    * **Loss** : Calculate the loss based on the competition values
    * **Inference** : Predict the output based on the competition values.

Depending on the phase (Training or Testing) Loss or Inference is used.

Inheritance Structure
****************************************

The Proto Y Architecture has a single base class that defines all steps and hooks
of the architecture.

.. autoclass:: prototorch.y.architectures.base.BaseYArchitecture

    **Steps**

    Components

    .. automethod:: init_components
    .. automethod:: components

    Backbone

    .. automethod:: init_backbone
    .. automethod:: backbone

    Comparison

    .. automethod:: init_comparison
    .. automethod:: comparison

    Competition

    .. automethod:: init_competition
    .. automethod:: competition

    Loss

    .. automethod:: init_loss
    .. automethod:: loss

    Inference

    .. automethod:: init_inference
    .. automethod:: inference

    **Hooks**

    Torchmetric

    .. automethod:: register_torchmetric

Hyperparameters
****************************************
Every model implemented with the Proto Y Architecture has a set of hyperparameters,
which is stored in the ``HyperParameters`` attribute of the architecture.
