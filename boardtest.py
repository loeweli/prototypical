from tensorboardX import SummaryWriter

import torch

import time

current_time = time.strftime("%Y-%m-%dT%H_%M_%S",time.localtime())
writer = SummaryWriter()
x = torch.FloatTensor([100])
y = torch.FloatTensor([100])

for epoch in range(100):
    x /= 1.5
    y /= 1.5
    loss = y - x
    writer.add_histogram("tttttttttt/x",x,epoch)
    writer.add_histogram("tttttttttt/y",y,epoch)

    writer.add_scalar('data/x',x ,epoch)
    writer.add_scalar('data/y',y,epoch)
    writer.add_scalar('data/loss',loss,epoch)
    writer.add_scalars('data/scalar_group',{'x':x,'y':y,'loss':loss},epoch)

    writer.add_text('zz/text', 'zz: this is epoch ' + str(epoch), epoch)
# export scalar data to JSON for external processing
# writer.export_scalars_to_json("./test.json")
writer.close()