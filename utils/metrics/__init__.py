try:
    from oct2py import octave as oc
    oc.addpath('utils/metrics/')
    ssim_index = oc.ssim_index
except:
    from utils.metrics.ssim_index import ssim
