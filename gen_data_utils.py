


def sigma_clip_maskonly(vals, previous_mask=None, sig=5):
    
    valcopy = vals.copy()
    if previous_mask is not None:
        valcopy[previous_mask==0] = np.nan
        sigma_val = np.nanstd(valcopy)
    else:
        sigma_val = np.nanstd(valcopy)
    
    abs_dev = np.abs(vals-np.nanmedian(valcopy))
    mask = (abs_dev < sig*sigma_val).astype(int)

    return mask

def compute_Neff(weights):
    N_eff = np.sum(weights)**2/np.sum(weights**2)

    return N_eff

def get_l2d(dimx, dimy, pixsize):
    lx = np.fft.fftfreq(dimx)*2
    ly = np.fft.fftfreq(dimy)*2
    lx = np.fft.ifftshift(lx)*(180*3600./pixsize)
    ly = np.fft.ifftshift(ly)*(180*3600./pixsize)
    ly, lx = np.meshgrid(ly, lx)
    l2d = np.sqrt(lx**2 + ly**2)
    
    return l2d

def generate_map_meshgrid(ra_cen, dec_cen, nside_deg, dimx, dimy):
    
    ra_range = np.linspace(ra_cen - 0.5*nside_deg, ra_cen + 0.5*nside_deg, dimx)
    dec_range = np.linspace(dec_cen - 0.5*nside_deg, dec_cen + 0.5*nside_deg, dimy)
    map_ra, map_dec = np.meshgrid(ra_range, dec_range)
    
    return map_ra, map_dec

def weighted_avg_and_std(values, weights):
    """
    Return the weighted average and standard deviation.

    values, weights -- Numpy ndarrays with the same shape.
    """
    average = np.average(values, weights=weights)
    # Fast and numerically precise:
    variance = np.average((values-average)**2, weights=weights)
    return average, np.sqrt(variance)

def dist(NAXIS):

    axis = np.linspace(-NAXIS/2+1, NAXIS/2, NAXIS)
    result = np.sqrt(axis**2 + axis[:,np.newaxis]**2)
    return np.roll(result, NAXIS/2+1, axis=(0,1))