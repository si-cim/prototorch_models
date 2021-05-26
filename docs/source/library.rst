.. Available Models

Models
========================================

Unsupervised Methods
-----------------------------------------
.. autoclass:: prototorch.models.unsupervised.KNN
   :members:

.. autoclass:: prototorch.models.unsupervised.NeuralGas
   :members:


Classical Learning Vector Quantization
-----------------------------------------
Original LVQ models by Kohonen.
These heuristic algorithms do not use gradient descent.

.. autoclass:: prototorch.models.lvq.LVQ1
   :members:
.. autoclass:: prototorch.models.lvq.LVQ21
   :members:

It is also possible to use the GLVQ structure as shown in [Sato&Yamada].
This allows the use of gradient descent methods.

.. autoclass:: prototorch.models.glvq.GLVQ1
   :members:
.. autoclass:: prototorch.models.glvq.GLVQ21
   :members:

Generalized Learning Vector Quantization
-----------------------------------------

.. autoclass:: prototorch.models.glvq.GLVQ
   :members:

.. autoclass:: prototorch.models.glvq.ImageGLVQ
   :members:

.. autoclass:: prototorch.models.glvq.SiameseGLVQ
   :members:

.. autoclass:: prototorch.models.glvq.GRLVQ
   :members:

.. autoclass:: prototorch.models.glvq.GMLVQ
   :members:

.. autoclass:: prototorch.models.glvq.LVQMLN
   :members:

Classification by Component
-----------------------------------------
.. autoclass:: prototorch.models.cbc.CBC
   :members:

.. autoclass:: prototorch.models.cbc.ImageCBC
   :members:

Visualization
========================================

.. automodule:: prototorch.models.vis
   :members:
   :undoc-members: