"""Makes 1 epoch fit to 1k steps in TensorBoard::

    from teebee import Teebee

    # Initialize a Teebee object with the per-epoch length (same with the
    # length of the loader) and the path where TensorBoard logs will be stored.
    tb = Teebee(len(loader), '/tmp/tb/run-20190219')

    for epoch in range(1000):
        # Let tb follow the current epoch.
        tb.epoch(epoch)

        # Report at the exact epoch.
        tb.scalar('lr', lr)

        for step, batch in enumerate(loader):
            ...

            # Let tb follow the current in-epoch step. It will return True
            # if logs should be reported at this step.
            if tb.step(step):
                # Calculate data to report in this block.
                tb.scalar('loss', loss.item())

"""
import logging

from tensorboardX import SummaryWriter

__all__ = ['Teebee']
__version__ = '0.1.2'


class Teebee:
    """A TensorBoard writer that tracks training epochs and steps. It reports
    1 epoch as 1k global steps in TensorBoard.

    It disallows inverted steps. Always increase epoch and step.

    Note:
        The name of "Teebee" came from simply "TB" which is an acronym for
        "TensorBoard".

    """
    __slots__ = (
        'epoch_length',
        'writer',
        '_epoch',
        '_step',
        '_global_step_increased',
    )

    def __init__(self, epoch_length: int, path: str = None):
        self.epoch_length = epoch_length

        if path is None:
            self.writer = None
        else:
            self.writer = SummaryWriter(path)

        self._epoch = -1
        self._step = -1
        self._global_step_increased = False

    def epoch(self, epoch: int):
        """Sets the current epoch and resets the step::

            for epoch in range(epochs):
                tb.epoch(epoch)
                ...

        Raises:
            ValueError: If the given epoch is less than the previous epoch.

        """
        if epoch < self._epoch:
            raise ValueError('already passed epoch: %d (new) < %d (old)'
                             '' % (epoch, self._epoch))

        self._epoch = epoch
        self._step = -1

    def step(self, step: int) -> bool:
        """Sets the current step in an epoch and returns whether it increases
        the global step.

            for step, inputs in enumerate(loader):
                ...
                if tb.step(step):
                    tb.scalar('loss', loss.item())

        The interval of steps is ``[0, epoch_length-1)``.

        If you report something when it returns ``False``, multiple points will
        be overlapped at the same global step::

            # DO NOT DO LIKE IT. ALWAYS USE "if tb.step(step):".
            tb.step(step)
            tb.scalar('loss', loss.item())

        Raises:
            ValueError: If the given step is greater than or equals to the
                epoch length.
            ValueError: If the given step is less than the previous step.

        """
        if step >= self.epoch_length:
            raise OverflowError('step is out of epoch length: '
                                '%d (step) >= %d (epoch length)'
                                '' % (step, self.epoch_length))

        if step < self._step:
            raise ValueError('already passed step: %d (new) < %d (old)'
                             '' % (step, self._step))

        if self._step == step:
            # When the step not changed, return the cached result.
            return self._global_step_increased

        if self._step == -1:
            # First step() per epoch always returns True.
            prev_global_step = -1
        else:
            prev_global_step = self.global_step()

        self._step = step

        self._global_step_increased = (self.global_step() != prev_global_step)
        return self._global_step_increased

    def global_step(self) -> int:
        """Calculates the current global step. 1k of global step means 1
        epoch.

        Raises:
            ValueError: If :meth:`epoch` has never been called.

        """
        if self._epoch < 0:
            raise ValueError('epoch never set')

        epoch_f = float(self._epoch)
        if self._step >= 0:
            epoch_f += (self._step+1) / self.epoch_length

        return int(epoch_f * 1000)

    @property
    def _l(self):
        # Initialize the logger on-demand.
        # https://fangpenlin.com/posts/2012/08/26/good-logging-practice-in-python/#do-not-get-logger-at-the-module-level-unless-disable_existing_loggers-is-false
        return logging.getLogger(__name__)

    def scalar(self, name: str, value: float):
        """Writes scalar data."""
        if self.writer is None:
            self._l.debug('[%d] %s: %.5f', self.global_step(), name, value)
        else:
            self.writer.add_scalar(name, value, self.global_step())

    def text(self, name: str, text: str):
        """Writes text data."""
        if self.writer is None:
            self._l.debug('[%d] %s:\n%s', self.global_step(), name, text)
        else:
            self.writer.add_text(name, text, self.global_step())
