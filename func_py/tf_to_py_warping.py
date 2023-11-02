import numpy as np
import torch
import torchvision.transforms.functional as TTF
import torch.nn as nn
import torch.nn.functional as F
def gather_nd_torch(params, indices, batch_dims=0):
    """ The same as tf.gather_nd.
    indices is an k-dimensional integer tensor, best thought of as a (k-1)-dimensional tensor of indices into params, where each element defines a slice of params:

    output[\\(i_0, ..., i_{k-2}\\)] = params[indices[\\(i_0, ..., i_{k-2}\\)]]

    Args:
        params (Tensor): "n" dimensions. shape: [x_0, x_1, x_2, ..., x_{n-1}]
        indices (Tensor): "k" dimensions. shape: [y_0,y_2,...,y_{k-2}, m]. m <= n.

    Returns: gathered Tensor.
        shape [y_0,y_2,...y_{k-2}] + params.shape[m:]

    """
    if isinstance(indices, torch.Tensor):
        indices = indices.numpy()
    else:
        if not isinstance(indices, np.array):
            raise ValueError(f'indices must be `torch.Tensor` or `numpy.array`. Got {type(indices)}')
    if batch_dims == 0:
        orig_shape = list(indices.shape)
        num_samples = int(np.prod(orig_shape[:-1]))
        m = orig_shape[-1]
        n = len(params.shape)

        if m <= n:
            out_shape = orig_shape[:-1] + list(params.shape[m:])
        else:
            raise ValueError(
                f'the last dimension of indices must less or equal to the rank of params. Got indices:{indices.shape}, params:{params.shape}. {m} > {n}'
            )
        indices = indices.reshape((num_samples, m)).transpose().tolist()
        output = params[indices]    # (num_samples, ...)
        return output.reshape(out_shape).contiguous()
    else:
        batch_shape = params.shape[:batch_dims]
        orig_indices_shape = list(indices.shape)
        orig_params_shape = list(params.shape)
        assert (
                batch_shape == indices.shape[:batch_dims]
        ), f'if batch_dims is not 0, then both "params" and "indices" have batch_dims leading batch dimensions that exactly match.'
        mbs = np.prod(batch_shape)
        if batch_dims != 1:
            params = params.reshape(mbs, *(params.shape[batch_dims:]))
            indices = indices.reshape(mbs, *(indices.shape[batch_dims:]))
        output = []
        for i in range(mbs):
            output.append(gather_nd_torch(params[i], indices[i], batch_dims=0))
        output = torch.stack(output, dim=0)
        output_shape = orig_indices_shape[:-1] + list(orig_params_shape[orig_indices_shape[-1]+batch_dims:])
        return output.reshape(*output_shape).contiguous()



def reciprocal_no_nan_pytorch(inputs):
    '''

    :param inputs: torch.tensor
    :return: 1 / inputs, nan,inf will be replaced by 0
    '''
    x = torch.reciprocal(inputs)
    result = torch.nan_to_num(input=x, nan=0,posinf=0,neginf=0)

    return result

def torch_rand_uniform(shape,dtype,minval = 0,maxval=10):
    '''
    Replacement of :
    tf.random.uniform(
      [], minval=min_relative_offset_h, maxval=max_relative_offset_h+1,
      dtype=tf.int32)

    '''

    return ((maxval-minval)* torch.rand(shape) + minval).type(dtype)

def torch_compat_v1_greater_equal(x,y,message = None):
    '''

    :param x: int,float,torch.tensor
    :param y: int,float,torch.tensor
    :param message: when x > y raise message

    :return:
    '''
    if x >= y:
        return
    else:
        if message != None:
            raise AssertionError(message)

        else:
            raise AssertionError("Plz check the inputs")
def torch_compat_v1_greater(x,y,message = None):
    '''

    :param x: int,float,torch.tensor
    :param y: int,float,torch.tensor
    :param message: when x > y raise message

    :return:
    '''
    if x > y:
        return
    else:
        if message != None:
            raise AssertionError(message)

        else:
            raise AssertionError("Plz check the inputs")
def torch_compat_v1_less_equal(x,y,message = None):
    '''

    :param x: int,float,torch.tensor
    :param y: int,float,torch.tensor
    :param message: when x > y raise message

    :return:
    '''
    if x <= y:
        return
    else:
        if message != None:
            raise AssertionError(message)

        else:
            raise AssertionError("Plz check the inputs")

def torch_ensure_shape(input:torch.tensor,shape:tuple):
    '''

    :param input: img -> torch.tensor [batch,channel,height, width]
    :param shape: tuple
    :return:
    '''
    if list(input.shape) != shape:
        raise AssertionError("The input's shape is not the shape you input")

    else:
        return input

class Callable_Concat(nn.Module):
    def __init__(self,dim):
        self.dim = dim

    def __call__(self,x):
        return torch.cat(x,dim=self.dim)

def torch_map_fn(func,data,dtype):
    get_Data = data.type(dtype)

    return func(get_Data)


def torch_checkpoint_manager(epochs,model_state,optimizer_state,loss,path_to_save):
    '''

    :param epochs: epoch times
    :param model_state: net.state_dict()
    :param optimizer_state: optimizer.state_dict()
    :param loss: loss list
    :param path_to_save: path to save the check point
    :return:
    '''
    torch.save({
        'epoch': epochs,
        'model_state_dict': model_state,
        'optimizer_state_dict': optimizer_state,
        'loss': loss,
    }, path_to_save)
class Checkpoint_Writer():
    def __init__(self, model, optimizer, epoch, loss, path):
        self.path = path

        self.di = {
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'loss': loss,
            }

    def save(self):
        return torch.save(self.di, self.path)

def unsorted_segment_sum(data, segment_ids, num_segments):
    """
    Computes the sum along segments of a tensor. Analogous to tf.unsorted_segment_sum.

    :param data: A tensor whose segments are to be summed.
    :param segment_ids: The segment indices tensor.
    :param num_segments: The number of segments.
    :return: A tensor of same data type as the data argument.
    """
    device = data.device
    assert all([i in data.shape for i in segment_ids.shape]), "segment_ids.shape should be a prefix of data.shape"

    # segment_ids is a 1-D tensor repeat it to have the same shape as data
    if len(segment_ids.shape) == 1:
        s = torch.prod(torch.tensor(data.shape[1:],device=device)).long()
        segment_ids = segment_ids.repeat_interleave(s).view(segment_ids.shape[0], *data.shape[1:])

    assert data.shape == segment_ids.shape, "data.shape and segment_ids.shape should be equal"

    shape = [num_segments] + list(data.shape[1:])
    tensor = torch.zeros(*shape,device=device).scatter_add(0, segment_ids, data.float())
    tensor = tensor.type(data.dtype)
    return tensor
if __name__ == '__main__':
    print(torch_rand_uniform([],dtype=torch.int32))