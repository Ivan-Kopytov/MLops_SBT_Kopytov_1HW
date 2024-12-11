import hydra
from omegaconf import DictConfig
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

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
    accelerator = cfg.train.accelerator
    devices = cfg.train.devices

    # Инициализируем DataModule и Model из параметров
    dm = RegressionDataModule(
        train_X=train_X,
        train_Y=train_Y,
        val_X=val_X,
        val_Y=val_Y,
        batch_size=batch_size
    )

    model = SimpleRegressionModel(lr=lr)

    # Настраиваем колбэк для сохранения чекпоинтов из конфигурации
    checkpoint_callback = ModelCheckpoint(
        monitor=cfg.train.checkpoint.monitor,
        dirpath=cfg.train.checkpoint.dirpath,
        filename=cfg.train.checkpoint.filename,
        save_top_k=cfg.train.checkpoint.save_top_k,
        mode=cfg.train.checkpoint.mode,
    )

    # Инициализируем тренер с колбэком
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator=accelerator,
        devices=devices,
        callbacks=[checkpoint_callback]
    )

    # Запускаем обучение
    trainer.fit(model, dm)

if __name__ == "__main__":
    main()
