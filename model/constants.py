from tensorboard.plugins.hparams import api as hp


hyperparameters = {"learning_rate": hp.HParam("learning_rate", hp.RealInterval(1e-5, 1e-3)),
                   "hidden_unit": hp.HParam("hidden_unit", hp.Discrete([16, 64])),
                   "batch_size": hp.HParam("batch_size", hp.Discrete([16, 32])),
                   "optimizer": hp.HParam("optimizer", hp.Discrete(["adamw"])),
                   "class_weights": hp.HParam("class_weights", hp.Discrete(["none", "balanced"])),
                   "dropout": hp.HParam("dropout", hp.Discrete([0.1, 0.4]))}

parameters = {
    "n_labels": 2,
}

metrics = [hp.Metric("accuracy", display_name="Accuracy"),
           hp.Metric("precision", display_name="Precision"),
           hp.Metric("recall", display_name="Recall"),
           hp.Metric("f1", display_name="F1 Score")]
