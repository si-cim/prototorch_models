"""prototorch.models test suite."""

import prototorch as pt
from prototorch.models.library import GLVQ


def test_glvq_model_build():
    hparams = GLVQ.HyperParameters(
        distribution=dict(num_classes=2, per_class=1),
        component_initializer=pt.initializers.RNCI(2),
    )

    model = GLVQ(hparams=hparams)
