import hydra
from omegaconf import DictConfig
import pytorch_lightning as pl

from data_module import RegressionDataModule
from model_module import SimpleRegressionModel

@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig):
    # Получаем параметры из cfg
    train_X = cfg.data.train_X
    train_Y = cfg.data.train_Y
    val_X = cfg.data.val_X
    val_Y = cfg.data.val_Y
    batch_size = cfg.data.batch_size

    lr = cfg.model.learning_rate

    max_epochs = cfg.train.max_epochs
    gpus = cfg.train.gpus

    # Инициализируем DataModule и Model из параметров
    dm = RegressionDataModule(
        train_X=train_X,
        train_Y=train_Y,
        val_X=val_X,
        val_Y=val_Y,
        batch_size=batch_size
    )

    model = SimpleRegressionModel(lr=lr)

    # # Инициализируем тренер
    # trainer = pl.Trainer(
    #     max_epochs=max_epochs,
    #     gpus=gpus  # Если используете lightning < 2.0, параметр gpus актуален, если Lightning >= 2.0, тогда используйте accelerator='gpu', devices=1
    # )
    trainer = pl.Trainer(
    max_epochs=max_epochs,
    accelerator="cpu",
    devices=1
    )
    # Запускаем обучение
    trainer.fit(model, dm)

if __name__ == "__main__":
    main()
