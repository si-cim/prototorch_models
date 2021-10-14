"""
CLCC Scheme

CLCC is a LVQ scheme containing 4 steps
- Components
- Latent Space
- Comparison
- Competition

"""
import pytorch_lightning as pl


class CLCCScheme(pl.LightningModule):
    def __init__(self, hparams) -> None:
        super().__init__()

        # Common Steps
        self.init_components(hparams)
        self.init_latent(hparams)
        self.init_comparison(hparams)
        self.init_competition(hparams)

        # Train Steps
        self.init_loss(hparams)

        # Inference Steps
        self.init_inference(hparams)

    # API
    def get_competion(self, batch, components):
        latent_batch, latent_components = self.latent(batch, components)
        # TODO: => Latent Hook
        comparison_tensor = self.comparison(latent_batch, latent_components)
        # TODO: => Comparison Hook
        return comparison_tensor

    def forward(self, batch):
        # TODO: manage different datatypes?
        components = self.components_layer()
        # TODO: => Component Hook
        comparison_tensor = self.get_competion(batch, components)
        # TODO: => Competition Hook
        return self.inference(comparison_tensor, components)

    def loss_forward(self, batch):
        # TODO: manage different datatypes?
        components = self.components_layer()
        # TODO: => Component Hook
        comparison_tensor = self.get_competion(batch, components)
        # TODO: => Competition Hook
        return self.loss(comparison_tensor, batch, components)

    # Empty Initialization
    # TODO: Type hints
    # TODO: Docs
    def init_components(self, hparams):
        ...

    def init_latent(self, hparams):
        ...

    def init_comparison(self, hparams):
        ...

    def init_competition(self, hparams):
        ...

    def init_loss(self, hparams):
        ...

    def init_inference(self, hparams):
        ...

    # Empty Steps
    # TODO: Type hints
    def components(self):
        """
        This step has no input.

        It returns the components.
        """
        raise NotImplementedError(
            "The components step has no reasonable default.")

    def latent(self, batch, components):
        """
        The latent step receives the data batch and the components.
        It can transform both by an arbitrary function.

        It returns the transformed batch and components, each of the same length as the original input.
        """
        return batch, components

    def comparison(self, batch, components):
        """
        Takes a batch of size N and the componentsset of size M.

        It returns an NxMxD tensor containing D (usually 1) pairwise comparison measures.
        """
        raise NotImplementedError(
            "The comparison step has no reasonable default.")

    def competition(self, comparisonmeasures, components):
        """
        Takes the tensor of comparison measures.

        Assigns a competition vector to each class.
        """
        raise NotImplementedError(
            "The competition step has no reasonable default.")

    def loss(self, comparisonmeasures, batch, components):
        """
        Takes the tensor of competition measures.

        Calculates a single loss value
        """
        raise NotImplementedError("The loss step has no reasonable default.")

    def inference(self, comparisonmeasures, components):
        """
        Takes the tensor of competition measures.

        Returns the inferred vector.
        """
        raise NotImplementedError(
            "The inference step has no reasonable default.")

    # Lightning Hooks
    def training_step(self, batch, batch_idx, optimizer_idx=None):
        return self.loss_forward(batch)

    def validation_step(self, batch, batch_idx):
        return self.loss_forward(batch)

    def test_step(self, batch, batch_idx):
        return self.loss_forward(batch)
