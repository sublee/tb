# teebee

```python
from teebee import TensorBoard

# Initialize a TensorBoard object with the per-epoch length
# and the path where TensorBoard logs will be stored.
tb = TensorBoard(len(loader), '/tmp/tb/run-20190219')

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
```
