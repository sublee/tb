"""Makes 1 epoch fit to 1k steps in TensorBoard::

    from tb import TB

    # Initialize tb with the per-epoch length and
    # the path where TensorBoard logs will be stored.
    tb = TB(len(loader), '/tmp/tb/run-20190219')

    for epoch in range(1000):
        # Write logs per epoch.
        tb[epoch].scalar('lr', lr)

        for i, batch in enumerate(loader):
            ...

            # Write logs per epoch/1000. Sometimes, writing would be skipped.
            w = tb[epoch:i]
            if w:
                # Calculate data to log in this block.
                # loss.item() is not cheap when using GPUs.
                w.scalar('loss', loss.item())

"""
from typing import Optional, Union

from tensorboardX import SummaryWriter


class Writer:
    """A wrapper of :class:`SummaryWriter` to write logs at the specific step.
    """

    def __init__(self, writer: Optional[SummaryWriter], step: int):
        self.writer = writer
        self.step = step

    def scalar(self, name: str, value: float):
        """Writes a scalar log."""
        if self.writer is None:
            print('[%d] %s: %.5f', self.step, name, value)
        else:
            self.writer.add_scalar(name, value, self.step)


class TB:
    """..."""

    def __init__(self, epoch_length: int, path: str = None):
        if path is None:
            self.writer = None
        else:
            self.writer = SummaryWriter(path)

        self.epoch_length = epoch_length
        self.last_step = -1

    def __getitem__(self, epoch_i: Union[int, slice]) -> Union[Writer, bool]:
        try:
            # [epoch:i]
            epoch: int = epoch_i.start
            i: int = epoch_i.stop
        except AttributeError:
            # [epoch]
            epoch: int = epoch_i
            i: int = None

        # Calculate the global step based on the given epoch and i.
        epoch_f = float(epoch)
        if i is not None:
            epoch_f += i / self.epoch_length
        step = int(epoch_f * 1000)

        if i is not None:
            # When i given, skip already progressed step.
            if self.last_step == step:
                return False
            self.last_step = step

        return Writer(self.writer, step)
