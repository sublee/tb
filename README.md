# teebee

```python
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
```
