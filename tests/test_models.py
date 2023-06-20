"""prototorch.models test suite."""

import prototorch.models


def test_glvq_model_build():
    model = prototorch.models.GLVQ(
        {"distribution": (3, 2)},
        prototypes_initializer=prototorch.initializers.RNCI(2),
    )


def test_glvq1_model_build():
    model = prototorch.models.GLVQ1(
        {"distribution": (3, 2)},
        prototypes_initializer=prototorch.initializers.RNCI(2),
    )


def test_glvq21_model_build():
    model = prototorch.models.GLVQ1(
        {"distribution": (3, 2)},
        prototypes_initializer=prototorch.initializers.RNCI(2),
    )


def test_gmlvq_model_build():
    model = prototorch.models.GMLVQ(
        {
            "distribution": (3, 2),
            "input_dim": 2,
            "latent_dim": 2,
        },
        prototypes_initializer=prototorch.initializers.RNCI(2),
    )


def test_grlvq_model_build():
    model = prototorch.models.GRLVQ(
        {
            "distribution": (3, 2),
            "input_dim": 2,
        },
        prototypes_initializer=prototorch.initializers.RNCI(2),
    )


def test_gtlvq_model_build():
    model = prototorch.models.GTLVQ(
        {
            "distribution": (3, 2),
            "input_dim": 4,
            "latent_dim": 2,
        },
        prototypes_initializer=prototorch.initializers.RNCI(2),
    )


def test_lgmlvq_model_build():
    model = prototorch.models.LGMLVQ(
        {
            "distribution": (3, 2),
            "input_dim": 4,
            "latent_dim": 2,
        },
        prototypes_initializer=prototorch.initializers.RNCI(2),
    )


def test_image_glvq_model_build():
    model = prototorch.models.ImageGLVQ(
        {"distribution": (3, 2)},
        prototypes_initializer=prototorch.initializers.RNCI(16),
    )


def test_image_gmlvq_model_build():
    model = prototorch.models.ImageGMLVQ(
        {
            "distribution": (3, 2),
            "input_dim": 16,
            "latent_dim": 2,
        },
        prototypes_initializer=prototorch.initializers.RNCI(16),
    )


def test_image_gtlvq_model_build():
    model = prototorch.models.ImageGMLVQ(
        {
            "distribution": (3, 2),
            "input_dim": 16,
            "latent_dim": 2,
        },
        prototypes_initializer=prototorch.initializers.RNCI(16),
    )


def test_siamese_glvq_model_build():
    model = prototorch.models.SiameseGLVQ(
        {"distribution": (3, 2)},
        prototypes_initializer=prototorch.initializers.RNCI(4),
    )


def test_siamese_gmlvq_model_build():
    model = prototorch.models.SiameseGMLVQ(
        {
            "distribution": (3, 2),
            "input_dim": 4,
            "latent_dim": 2,
        },
        prototypes_initializer=prototorch.initializers.RNCI(4),
    )


def test_siamese_gtlvq_model_build():
    model = prototorch.models.SiameseGTLVQ(
        {
            "distribution": (3, 2),
            "input_dim": 4,
            "latent_dim": 2,
        },
        prototypes_initializer=prototorch.initializers.RNCI(4),
    )


def test_knn_model_build():
    train_ds = prototorch.datasets.Iris(dims=[0, 2])
    model = prototorch.models.KNN(dict(k=3), data=train_ds)


def test_lvq1_model_build():
    model = prototorch.models.LVQ1(
        {"distribution": (3, 2)},
        prototypes_initializer=prototorch.initializers.RNCI(2),
    )


def test_lvq21_model_build():
    model = prototorch.models.LVQ21(
        {"distribution": (3, 2)},
        prototypes_initializer=prototorch.initializers.RNCI(2),
    )


def test_median_lvq_model_build():
    model = prototorch.models.MedianLVQ(
        {"distribution": (3, 2)},
        prototypes_initializer=prototorch.initializers.RNCI(2),
    )


def test_celvq_model_build():
    model = prototorch.models.CELVQ(
        {"distribution": (3, 2)},
        prototypes_initializer=prototorch.initializers.RNCI(2),
    )


def test_rslvq_model_build():
    model = prototorch.models.RSLVQ(
        {"distribution": (3, 2)},
        prototypes_initializer=prototorch.initializers.RNCI(2),
    )


def test_slvq_model_build():
    model = prototorch.models.SLVQ(
        {"distribution": (3, 2)},
        prototypes_initializer=prototorch.initializers.RNCI(2),
    )


def test_growing_neural_gas_model_build():
    model = prototorch.models.GrowingNeuralGas(
        {"num_prototypes": 5},
        prototypes_initializer=prototorch.initializers.RNCI(2),
    )


def test_kohonen_som_model_build():
    model = prototorch.models.KohonenSOM(
        {"shape": (3, 2)},
        prototypes_initializer=prototorch.initializers.RNCI(2),
    )


def test_neural_gas_model_build():
    model = prototorch.models.NeuralGas(
        {"num_prototypes": 5},
        prototypes_initializer=prototorch.initializers.RNCI(2),
    )
