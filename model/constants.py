from tensorboard.plugins.hparams import api as hp


hyperparameters = {"learning_rate": hp.HParam("learning_rate", hp.RealInterval(1e-3, 1e-1)),
                   "hidden_unit": hp.HParam("hidden_unit", hp.Discrete([8, 16, 32, 64, 128])),
                   "batch_size": hp.HParam("batch_size", hp.Discrete([16, 32, 64])),
                   "optimizer": hp.HParam("optimizer", hp.Discrete(["adam", "adamw", "sgd"])),
                   "class_weights": hp.HParam("class_weights", hp.Discrete(["none", "balanced"])),
                   "dropout": hp.HParam("dropout", hp.Discrete([0.1, 0.2, 0.3]))}

parameters = {
    "vocabulary_size": 5001,
    "embedding_dim": 32,
    "n_labels": 2,
    "epochs": 3,
}

metrics = [hp.Metric("accuracy", display_name="Accuracy"),
           hp.Metric("precision", display_name="Precision"),
           hp.Metric("recall", display_name="Recall"),
           hp.Metric("f1", display_name="F1 Score")]