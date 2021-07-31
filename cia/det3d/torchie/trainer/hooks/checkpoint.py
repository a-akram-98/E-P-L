from ..utils import master_only
from .hook import Hook


class CheckpointHook(Hook):
    def __init__(self, interval=1, save_optimizer=True, out_dir=None, **kwargs):
        '''
            checkpoint_config = dict(interval=1)
        '''
        self.interval = interval
        self.save_optimizer = save_optimizer
        self.out_dir = out_dir
        self.args = kwargs

    @master_only
    def after_train_epoch(self, trainer):
        if not self.every_n_epochs(trainer, 1):
            return

        if not self.out_dir:  # True
            self.out_dir = trainer.cfg.cp_dir
        trainer.AANET_model.saveCheckpoint(trainer.epoch() , trainer.inner_iter())
        trainer.save_checkpoint(self.out_dir, save_optimizer=self.save_optimizer, **self.args)
