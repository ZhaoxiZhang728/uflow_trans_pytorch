
import torch
from utils.uflow_utils import resize,census_transform,soft_hamming, zero_mask_border
from utils import uflow_plotting

def l1(x):
  return torch.abs(x + 1e-6)


def robust_l1(x):
  """Robust L1 metric."""
  return (x**2 + 0.001**2)**0.5


def abs_robust_loss(diff, eps=0.01, q=0.4):
  """The so-called robust loss used by DDFlow."""
  return torch.pow((torch.abs(diff) + eps), q)


def _avg_pool3x3(x):
  return torch.nn.functional.avg_pool2d(input = x, kernel_size = (3,3),stride = 1,padding = 0)

def image_grads(image_batch, stride=1): # twisit
  image_batch_gh = image_batch[:, :, stride:, :] - image_batch[:, :, :-stride, :]
  image_batch_gw = image_batch[:, :, :, stride:] - image_batch[:, :, :, :-stride]
  return image_batch_gh, image_batch_gw

def get_distance_metric_fns(distance_metrics):
  """Returns a dictionary of distance metrics."""
  output = {}
  for key, distance_metric in distance_metrics.items():
    if distance_metric == 'l1':
      output[key] = l1
    elif distance_metric == 'robust_l1':
      output[key] = robust_l1
    elif distance_metric == 'ddflow':
      output[key] = abs_robust_loss
    else:
      raise ValueError('Unknown loss function')
  return output

def weighted_ssim(x, y, weight, c1=float('inf'), c2=9e-6, weight_epsilon=0.01):
  """Computes a weighted structured image similarity measure.

  See https://en.wikipedia.org/wiki/Structural_similarity#Algorithm. The only
  difference here is that not all pixels are weighted equally when calculating
  the moments - they are weighted by a weight function.

  Args:
    x: A tf.Tensor representing a batch of images, of shape [B, H, W, C].
    y: A tf.Tensor representing a batch of images, of shape [B, H, W, C].
    weight: A tf.Tensor of shape [B, H, W], representing the weight of each
      pixel in both images when we come to calculate moments (means and
      correlations).
    c1: A floating point number, regularizes division by zero of the means.
    c2: A floating point number, regularizes division by zero of the second
      moments.
    weight_epsilon: A floating point number, used to regularize division by the
      weight.

  Returns:
    A tuple of two tf.Tensors. First, of shape [B, H-2, W-2, C], is scalar
    similarity loss oer pixel per channel, and the second, of shape
    [B, H-2. W-2, 1], is the average pooled `weight`. It is needed so that we
    know how much to weigh each pixel in the first tensor. For example, if
    `'weight` was very small in some area of the images, the first tensor will
    still assign a loss to these pixels, but we shouldn't take the result too
    seriously.
  """
  if c1 == float('inf') and c2 == float('inf'):
    raise ValueError('Both c1 and c2 are infinite, SSIM loss is zero. This is '
                     'likely unintended.')
  weight = torch.unsqueeze(weight, -1)
  average_pooled_weight = _avg_pool3x3(weight)
  weight_plus_epsilon = weight + weight_epsilon
  inverse_average_pooled_weight = 1.0 / (average_pooled_weight + weight_epsilon)

  def weighted_avg_pool3x3(z):
    wighted_avg = _avg_pool3x3(z * weight_plus_epsilon)
    return wighted_avg * inverse_average_pooled_weight

  mu_x = weighted_avg_pool3x3(x)
  mu_y = weighted_avg_pool3x3(y)
  sigma_x = weighted_avg_pool3x3(x**2) - mu_x**2
  sigma_y = weighted_avg_pool3x3(y**2) - mu_y**2
  sigma_xy = weighted_avg_pool3x3(x * y) - mu_x * mu_y
  if c1 == float('inf'):
    ssim_n = (2 * sigma_xy + c2)
    ssim_d = (sigma_x + sigma_y + c2)
  elif c2 == float('inf'):
    ssim_n = 2 * mu_x * mu_y + c1
    ssim_d = mu_x**2 + mu_y**2 + c1
  else:
    ssim_n = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    ssim_d = (mu_x**2 + mu_y**2 + c1) * (sigma_x + sigma_y + c2)
  result = ssim_n / ssim_d
  return torch.clamp((1 - result) / 2, 0, 1), average_pooled_weight


def census_loss(image_a_bhw3,
                image_b_bhw3,
                mask_bhw3,
                patch_size=7,
                distance_metric_fn=abs_robust_loss):
  """Compares the similarity of the census transform of two images."""
  census_image_a_bhwk = census_transform(image_a_bhw3, patch_size)
  census_image_b_bhwk = census_transform(image_b_bhw3, patch_size)

  hamming_bhw1 = soft_hamming(census_image_a_bhwk, census_image_b_bhwk)

  # Set borders of mask to zero to ignore edge effects.
  padded_mask_bhw3 = zero_mask_border(mask_bhw3, patch_size)
  diff = distance_metric_fn(hamming_bhw1)
  diff *= padded_mask_bhw3
  diff_sum = torch.sum(input=diff)
  #padded_mask_bhw3_no_grad = padded_mask_bhw3.clone().detach()
  #padded_mask_bhw3_no_grad.requires_grad = False
  loss_mean = diff_sum / (
      torch.sum(input=padded_mask_bhw3.clone().detach() + 1e-6))
  return loss_mean

def compute_loss(
    weights,
    images,
    flows,
    warps,
    valid_warp_masks,
    not_occluded_masks,
    fb_sq_diff,
    fb_sum_sq,
    warped_images,
    only_forward=False,
    selfsup_transform_fns=None,
    fb_sigma_teacher=0.003,
    fb_sigma_student=0.03,
    plot_dir=None,
    distance_metrics=None,
    smoothness_edge_weighting='gaussian',
    stop_gradient_mask=True,
    selfsup_mask='gaussian',
    ground_truth_occlusions=None,
    smoothness_at_level=2,
):
  """Compute UFlow losses."""
  if distance_metrics is None:
    distance_metrics = {
        'photo': 'robust_l1',
        'census': 'ddflow',
    }
  distance_metric_fns = get_distance_metric_fns(distance_metrics)
  losses = dict()
  for key in weights:
    if key not in ['edge_constant']:
      losses[key] = 0.0

  compute_loss_for_these_flows = ['augmented-student']
  # Count number of non self-sup pairs, for which we will apply the losses.
  num_pairs = sum(
      [1.0 for (i, j, c) in warps if c in compute_loss_for_these_flows])

  # Iterate over image pairs.
  for key in warps:
    i, j, c = key

    if c not in compute_loss_for_these_flows or (only_forward and i > j):
      continue

    if ground_truth_occlusions is None:
      if stop_gradient_mask:
          with torch.no_grad():
            mask_level0 = not_occluded_masks[key][0] *valid_warp_masks[key][0]
      else:
          mask_level0 = not_occluded_masks[key][0] * valid_warp_masks[key][0]
    else:
      # For using ground truth mask
      if i > j:
        continue
      ground_truth_occlusions = 1.0 - ground_truth_occlusions.type(torch.float32)
      with torch.no_grad():
        mask_level0 = (ground_truth_occlusions * valid_warp_masks[key][0])
      height, width = valid_warp_masks[key][1].shape[-2:]

    if 'photo' in weights:
      error = distance_metric_fns['photo'](images[i] - warped_images[key])
      losses['photo'] += (
          weights['photo'] * torch.sum(input=mask_level0 * error) /
          (torch.sum(input=mask_level0) + 1e-16) / num_pairs)

    if 'smooth2' in weights or 'smooth1' in weights:

      edge_constant = 0.0
      if 'edge_constant' in weights:
        edge_constant = weights['edge_constant']

      abs_fn = None
      if smoothness_edge_weighting == 'gaussian':
        abs_fn = lambda x: x**2
      elif smoothness_edge_weighting == 'exponential':
        abs_fn = abs

      # Compute image gradients and sum them up to match the receptive field
      # of the flow gradients, which are computed at 1/4 resolution.
      images_level0 = images[i]
      height, width = images_level0.shape[-2:] #original
      # Resize two times for a smoother result.
      images_level1 = resize(
          images_level0, int(height) // 2, int(width) // 2, is_flow=False)
      images_level2 = resize(
          images_level1, int(height) // 4, int(width) // 4, is_flow=False)
      images_at_level = [images_level0, images_level1, images_level2]

      if 'smooth1' in weights:

        img_gx, img_gy = image_grads(images_at_level[smoothness_at_level])
        weights_x = torch.exp(-torch.mean(
            input=(abs_fn(edge_constant * img_gx)),
            dim=1, #original axis = -1
            keepdim=True))
        weights_y = torch.exp(-torch.mean(
            input=(abs_fn(edge_constant * img_gy)),
            dim=1,
            keepdim=True))

        # Compute second derivatives of the predicted smoothness.
        flow_gx, flow_gy = image_grads(flows[key][smoothness_at_level])

        # Compute weighted smoothness
        losses['smooth1'] += (
            weights['smooth1'] *
            (torch.mean(input=weights_x * robust_l1(flow_gx)) +
             torch.mean(input=weights_y * robust_l1(flow_gy))) / 2. /
            num_pairs)

        if plot_dir is not None:
          uflow_plotting.plot_smoothness(key, images, weights_x, weights_y,
                                         robust_l1(flow_gx), robust_l1(flow_gy),
                                         flows, plot_dir)

      if 'smooth2' in weights:

        img_gx, img_gy = image_grads(
            images_at_level[smoothness_at_level], stride=2)

        weights_xx = torch.exp(-torch.mean(
            input=(abs_fn(edge_constant * img_gx)),
            dim=1, #original: axis = -1
            keepdim=True))
        weights_yy = torch.exp(-torch.mean(
            input=(abs_fn(edge_constant * img_gy)),
            dim=1, #original: axis = -1
            keepdim=True))

        # Compute second derivatives of the predicted smoothness.
        flow_gx, flow_gy = image_grads(flows[key][smoothness_at_level])

        flow_gxx, unused_flow_gxy = image_grads(flow_gx)
        unused_flow_gyx, flow_gyy = image_grads(flow_gy)


        # Compute weighted smoothness
        losses['smooth2'] += (
            weights['smooth2'] *
            (torch.mean(input=weights_xx * robust_l1(flow_gxx)) +
             torch.mean(input=weights_yy * robust_l1(flow_gyy))) /
            2. / num_pairs)

        if plot_dir is not None:
          uflow_plotting.plot_smoothness(key, images, weights_xx, weights_yy,
                                         robust_l1(flow_gxx),
                                         robust_l1(flow_gyy), flows, plot_dir)

    if 'ssim' in weights:
      ssim_error, avg_weight = weighted_ssim(warped_images[key], images[i],
                                             torch.squeeze(mask_level0, dim=-1))

      losses['ssim'] += weights['ssim'] * (
          torch.sum(input=ssim_error * avg_weight) /
          (torch.sum(input=avg_weight) + 1e-16) / num_pairs)

    if 'census' in weights:
      losses['census'] += weights['census'] * census_loss(
          images[i],
          warped_images[key],
          mask_level0,
          distance_metric_fn=distance_metric_fns['census']) / num_pairs

    if 'selfsup' in weights:
      assert selfsup_transform_fns is not None
      _, _ ,h, w = list(flows[key][2].shape)
      teacher_flow = flows[(i, j, 'original-teacher')][2]
      student_flow = flows[(i, j, 'transformed-student')][2]
      teacher_flow = selfsup_transform_fns[2](
          teacher_flow, i_or_ij=(i, j), is_flow=True)
      if selfsup_mask == 'gaussian':
        student_fb_consistency = torch.exp(
            -fb_sq_diff[(i, j, 'transformed-student')][2] /
            (fb_sigma_student**2 * (h**2 + w**2)))
        teacher_fb_consistency = torch.exp(
            -fb_sq_diff[(i, j, 'original-teacher')][2] / (fb_sigma_teacher**2 *
                                                          (h**2 + w**2)))
      elif selfsup_mask == 'advection':
        student_fb_consistency = not_occluded_masks[(i, j,
                                                     'transformed-student')][2]
        teacher_fb_consistency = not_occluded_masks[(i, j,
                                                     'original-teacher')][2]
      elif selfsup_mask == 'ddflow':
        threshold_student = 0.01 * (fb_sum_sq[
            (i, j, 'transformed-student')][2]) + 0.5
        threshold_teacher = 0.01 * (fb_sum_sq[
            (i, j, 'original-teacher')][2]) + 0.5
        student_fb_consistency = (fb_sq_diff[(i, j, 'transformed-student')][2] < threshold_student).type(torch.float32)
        teacher_fb_consistency = (fb_sq_diff[(i, j, 'original-teacher')][2] < threshold_teacher).type(torch.float32)
      else:
        raise ValueError('Unknown selfsup_mask', selfsup_mask)

      student_mask = 1. - (
          student_fb_consistency *
          valid_warp_masks[(i, j, 'transformed-student')][2])
      teacher_mask = (
          teacher_fb_consistency *
          valid_warp_masks[(i, j, 'original-teacher')][2])
      teacher_mask = selfsup_transform_fns[2](
          teacher_mask, i_or_ij=(i, j), is_flow=False)


      error = robust_l1(teacher_flow.clone().detach()- student_flow.clone().detach())
      with torch.no_grad():
        mask = teacher_mask * student_mask
      losses['selfsup'] += (
          weights['selfsup'] * torch.sum(input=mask * error) /
          (torch.sum(input=torch.ones_like(mask)) + 1e-16) / num_pairs)
      if plot_dir is not None:
        uflow_plotting.plot_selfsup(key, images, flows, teacher_flow,
                                    student_flow, error, teacher_mask,
                                    student_mask, mask, selfsup_transform_fns,
                                    plot_dir)

  losses['total'] = sum(losses.values())

  return losses



def supervised_loss(weights, ground_truth_flow, ground_truth_valid,
                    predicted_flows):
  """Returns a supervised l1 loss when ground-truth flow is provided."""
  losses = {}
  predicted_flow = predicted_flows[(0, 1, 'augmented')][0]
  # ground truth flow is given from image 0 to image 1
  #predicted_flow = predicted_flows[(0, 1, 'augmented')][0]
  # resize flow to match ground truth (only changes resolution if ground truth
  # flow was not resized during loading (resize_gt_flow=False)
  _, _, height, width = list(ground_truth_flow.shape)
  predicted_flow = resize(predicted_flow, height, width, is_flow=True)
  device = ground_truth_flow.device
  # compute error/loss metric
  error = robust_l1(ground_truth_flow - predicted_flow)
  if ground_truth_valid is None:
    b, _ ,h, w = list(ground_truth_flow.shape)
    ground_truth_valid = torch.ones(size = (b, 1, h, w), dtype=torch.float32,device=device)
  losses['supervision'] = (
      weights['supervision'] *
      torch.sum(input=ground_truth_valid * error) /
      (torch.sum(input=ground_truth_valid) + 1e-16))
  losses['total'] = losses['supervision']

  return losses

