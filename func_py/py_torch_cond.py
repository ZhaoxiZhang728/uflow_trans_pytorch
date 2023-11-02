import torch.nn as nn
import torch
class Cond(nn.Module):
    def __init__(self, pred, True_fn, False_fn):
        super(Cond, self).__init__()
        self.pred = bool(pred)
        self.True_fn = True_fn
        self.False_fn = False_fn


    def forward(self):
        if self.pred:
            result = self.True_fn()
        else:
            result = self.False_fn()

        return result

def torch_cond(pred, True_fn, False_fn):
    cond = Cond(pred, True_fn, False_fn)

    return cond.forward()
if __name__ == "__main__":
    x = 1
    r = 10
    func1 = lambda: x + 1
    func2 =lambda: x-1

    cond = Cond()
    result = cond(r,func1,func2)

    print(result)