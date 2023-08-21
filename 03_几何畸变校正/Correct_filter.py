import ee
import math
import numpy as np
from functools import partial

# -----------------------------------------------地形校正-----------------------------------------
def slope_correction(collection, elevation, model, buffer=0):
    '''This function applies the slope correction on a collection of Sentinel-1 data

       :param collection: ee.Collection or ee.Image of Sentinel-1
       :param elevation: ee.Image of DEM
       :param model: model to be applied (volume/surface)
       :param buffer: buffer in meters for layover/shadow amsk
       #
       :returns: ee.Image
    '''

    def _volumetric_model_SCF(theta_iRad, alpha_rRad):
        '''Code for calculation of volumetric model SCF
            体积模型
        :param theta_iRad: ee.Image of incidence angle in radians
        :param alpha_rRad: ee.Image of slope steepness in range

        :returns: ee.Image
        '''

        # create a 90 degree image in radians
        ninetyRad = ee.Image.constant(90).multiply(np.pi / 180)

        # model
        nominator = (ninetyRad.subtract(theta_iRad).add(alpha_rRad)).tan()
        denominator = (ninetyRad.subtract(theta_iRad)).tan()
        return nominator.divide(denominator)

    def _surface_model_SCF(theta_iRad, alpha_rRad, alpha_azRad):
        '''Code for calculation of direct model SCF
            表面模型
        :param theta_iRad: ee.Image of incidence angle in radians
        :param alpha_rRad: ee.Image of slope steepness in range
        :param alpha_azRad: ee.Image of slope steepness in azimuth

        :returns: ee.Image
        '''

        # create a 90 degree image in radians
        ninetyRad = ee.Image.constant(90).multiply(np.pi / 180)

        # model
        nominator = (ninetyRad.subtract(theta_iRad)).cos()
        denominator = (alpha_azRad.cos().multiply(
            (ninetyRad.subtract(theta_iRad).add(alpha_rRad)).cos()))

        return nominator.divide(denominator)

    def _erode(image, distance):
        '''Buffer function for raster
            腐蚀算法，输入的图像需要额外的缓冲
        :param image: ee.Image that shoudl be buffered
        :param distance: distance of buffer in meters

        :returns: ee.Image
        '''

        d = (image.Not().unmask(1).fastDistanceTransform(30).sqrt().multiply(
            ee.Image.pixelArea().sqrt()))

        return image.updateMask(d.gt(distance))

    def _masking(alpha_rRad, theta_iRad, buffer):
        '''Masking of layover and shadow
            获取几何畸变区域
        :param alpha_rRad: ee.Image of slope steepness in range
        :param theta_iRad: ee.Image of incidence angle in radians
        :param buffer: buffer in meters

        :returns: ee.Image
        '''
        # layover, where slope > radar viewing angle
        layover = alpha_rRad.lt(theta_iRad).rename('layover')

        # shadow
        ninetyRad = ee.Image.constant(90).multiply(np.pi / 180)
        shadow = alpha_rRad.gt(
            ee.Image.constant(-1).multiply(
                ninetyRad.subtract(theta_iRad))).rename('shadow')

        # add buffer to layover and shadow
        if buffer > 0:
            layover = _erode(layover, buffer)
            shadow = _erode(shadow, buffer)

        # combine layover and shadow
        no_data_mask = layover.And(shadow).rename('no_data_mask')

        return layover.addBands(shadow).addBands(no_data_mask)

    def _correct(image):
        '''This function applies the slope correction and adds layover and shadow masks

        '''

        # get the image geometry and projection
        geom = image.geometry()
        proj = image.select(1).projection()

        # calculate the look direction
        heading = (ee.Terrain.aspect(image.select('angle')).reduceRegion(
            ee.Reducer.mean(), geom, 1000).get('aspect'))

        # Sigma0 to Power of input image，Sigma0是指雷达回波信号的强度
        sigma0Pow = ee.Image.constant(10).pow(image.divide(10.0))

        # the numbering follows the article chapters
        # 2.1.1 Radar geometry
        theta_iRad = image.select('angle').multiply(np.pi / 180)
        phi_iRad = ee.Image.constant(heading).multiply(np.pi / 180)

        # 2.1.2 Terrain geometry
        alpha_sRad = ee.Terrain.slope(elevation).select('slope').multiply(
            np.pi / 180).setDefaultProjection(proj).clip(geom)
        phi_sRad = ee.Terrain.aspect(elevation).select('aspect').multiply(
            np.pi / 180).setDefaultProjection(proj).clip(geom)

        # we get the height, for export
        height = elevation.setDefaultProjection(proj).clip(geom)

        # 2.1.3 Model geometry
        # reduce to 3 angle
        phi_rRad = phi_iRad.subtract(phi_sRad)

        # slope steepness in range (eq. 2)
        alpha_rRad = (alpha_sRad.tan().multiply(phi_rRad.cos())).atan()

        # slope steepness in azimuth (eq 3)
        alpha_azRad = (alpha_sRad.tan().multiply(phi_rRad.sin())).atan()

        # local incidence angle (eq. 4)
        theta_liaRad = (alpha_azRad.cos().multiply(
            (theta_iRad.subtract(alpha_rRad)).cos())).acos()
        theta_liaDeg = theta_liaRad.multiply(180 / np.pi)

        # 2.2
        # Gamma_nought
        gamma0 = sigma0Pow.divide(theta_iRad.cos())
        gamma0dB = ee.Image.constant(10).multiply(gamma0.log10()).select(
            ['VV', 'VH'], ['VV_gamma0', 'VH_gamma0'])
        ratio_gamma = (gamma0dB.select('VV_gamma0').subtract(
            gamma0dB.select('VH_gamma0')).rename('ratio_gamma0'))

        if model == 'volume':
            scf = _volumetric_model_SCF(theta_iRad, alpha_rRad)

        if model == 'surface':
            scf = _surface_model_SCF(theta_iRad, alpha_rRad, alpha_azRad)

        # apply model for Gamm0_f
        gamma0_flat = gamma0.divide(scf)
        gamma0_flatDB = (ee.Image.constant(10).multiply(
            gamma0_flat.log10()).select(['VV', 'VH'],
                                        ['VV_gamma0flat', 'VH_gamma0flat']))

        masks = _masking(alpha_rRad, theta_iRad, buffer)

        # calculate the ratio for RGB vis
        ratio_flat = (gamma0_flatDB.select('VV_gamma0flat').subtract(
            gamma0_flatDB.select('VH_gamma0flat')).rename('ratio_gamma0flat'))

        return (image.rename(['VV_sigma0', 'VH_sigma0', 'incAngle'])
                .addBands(gamma0dB)
                .addBands(ratio_gamma)
                .addBands(gamma0_flatDB)
                .addBands(ratio_flat)
                .addBands(alpha_rRad.rename('alpha_rRad'))
                .addBands(alpha_azRad.rename('alpha_azRad'))
                .addBands(phi_sRad.rename('aspect'))
                .addBands(alpha_sRad.rename('slope'))
                .addBands(theta_iRad.rename('theta_iRad'))
                .addBands(theta_liaRad.rename('theta_liaRad'))
                .addBands(masks).addBands(height.rename('elevation')))

    # run and return correction
    if type(collection) == ee.imagecollection.ImageCollection:
        return collection.map(_correct)
    elif type(collection) == ee.image.Image:
        return _correct(collection)
    else:
        print('Check input type, only image collection or image can be input')

# -----------------------------------------------图像滤波-----------------------------------------
def boxcar(KERNEL_SIZE):
    def wrap(image):
        bandNames = image.bandNames().remove('angle')
        # Define a boxcar kernel
        kernel = ee.Kernel.square((KERNEL_SIZE / 2), units='pixels', normalize=True)
        # Apply boxcar
        output = image.select(bandNames).convolve(kernel).rename(bandNames)
        return image.addBands(output, None, True)

    return wrap

def leefilter(KERNEL_SIZE):
    def wrap(image):
        bandNames = image.bandNames().remove('angle')
        # S1-GRD images are multilooked 5 times in range
        enl = 5
        # Compute the speckle standard deviation
        eta = 1.0 / math.sqrt(enl)
        eta = ee.Image.constant(eta)
        # MMSE estimator
        # Neighbourhood mean and variance
        oneImg = ee.Image.constant(1)
        # Estimate stats
        reducers = ee.Reducer.mean().combine(
            reducer2=ee.Reducer.variance()
            , sharedInputs=True
        )
        stats = (image.select(bandNames).reduceNeighborhood(
            reducer=reducers
            , kernel=ee.Kernel.square(KERNEL_SIZE / 2, 'pixels')
            , optimization='window'))
        meanBand = bandNames.map(lambda bandName: ee.String(bandName).cat('_mean'))
        varBand = bandNames.map(lambda bandName: ee.String(bandName).cat('_variance'))

        z_bar = stats.select(meanBand)
        varz = stats.select(varBand)
        # Estimate weight
        varx = (varz.subtract(z_bar.pow(2).multiply(eta.pow(2)))).divide(oneImg.add(eta.pow(2)))
        b = varx.divide(varz)
        # if b is negative set it to zero
        new_b = b.where(b.lt(0), 0)
        output = oneImg.subtract(new_b).multiply(z_bar.abs()).add(new_b.multiply(image.select(bandNames)))
        output = output.rename(bandNames)
        return image.addBands(output, None, True)

    return wrap

def gammamap(KERNEL_SIZE):
    def wrap(image):
        enl = 5
        bandNames = image.bandNames().remove('angle')
        # local mean
        reducers = ee.Reducer.mean().combine( \
            reducer2=ee.Reducer.stdDev(), \
            sharedInputs=True
        )
        stats = (image.select(bandNames).reduceNeighborhood( \
            reducer=reducers, \
            kernel=ee.Kernel.square(KERNEL_SIZE / 2, 'pixels'), \
            optimization='window'))
        meanBand = bandNames.map(lambda bandName: ee.String(bandName).cat('_mean'))
        stdDevBand = bandNames.map(lambda bandName: ee.String(bandName).cat('_stdDev'))
        z = stats.select(meanBand)
        sigz = stats.select(stdDevBand)
        # local observed coefficient of variation
        ci = sigz.divide(z)
        # noise coefficient of variation (or noise sigma)
        cu = 1.0 / math.sqrt(enl)
        # threshold for the observed coefficient of variation
        cmax = math.sqrt(2.0) * cu
        cu = ee.Image.constant(cu)
        cmax = ee.Image.constant(cmax)
        enlImg = ee.Image.constant(enl)
        oneImg = ee.Image.constant(1)
        twoImg = ee.Image.constant(2)
        alpha = oneImg.add(cu.pow(2)).divide(ci.pow(2).subtract(cu.pow(2)))
        # Implements the Gamma MAP filter described in equation 11 in Lopez et al. 1990
        q = image.select(bandNames).expression('z**2 * (z * alpha - enl - 1)**2 + 4 * alpha * enl * b() * z',
                                               {'z': z, 'alpha': alpha, 'enl': enl})
        rHat = z.multiply(alpha.subtract(enlImg).subtract(oneImg)).add(q.sqrt()).divide(twoImg.multiply(alpha))
        # if ci <= cu then its a homogenous region ->> boxcar filter
        zHat = (z.updateMask(ci.lte(cu))).rename(bandNames)
        # if cmax > ci > cu then its a textured medium ->> apply Gamma MAP filter
        rHat = (rHat.updateMask(ci.gt(cu)).updateMask(ci.lt(cmax))).rename(bandNames)
        # ci>cmax then its strong signal ->> retain
        x = image.select(bandNames).updateMask(ci.gte(cmax)).rename(bandNames)
        # Merge
        output = ee.ImageCollection([zHat, rHat, x]).sum()
        return image.addBands(output, None, True)

    return wrap

def RefinedLee(image):
    bandNames = image.bandNames().remove('angle')

    def inner(b):
        img = image.select([b]);

        # img must be linear, i.e. not in dB!
        # Set up 3x3 kernels
        weights3 = ee.List.repeat(ee.List.repeat(1, 3), 3);
        kernel3 = ee.Kernel.fixed(3, 3, weights3, 1, 1, False);

        mean3 = img.reduceNeighborhood(ee.Reducer.mean(), kernel3);
        variance3 = img.reduceNeighborhood(ee.Reducer.variance(), kernel3);

        # Use a sample of the 3x3 windows inside a 7x7 windows to determine gradients and directions
        sample_weights = ee.List(
            [[0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 1, 0, 1, 0], [0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 1, 0, 1, 0],
             [0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 1, 0, 1, 0], [0, 0, 0, 0, 0, 0, 0]]);

        sample_kernel = ee.Kernel.fixed(7, 7, sample_weights, 3, 3, False);

        # Calculate mean and variance for the sampled windows and store as 9 bands
        sample_mean = mean3.neighborhoodToBands(sample_kernel);
        sample_var = variance3.neighborhoodToBands(sample_kernel);

        # Determine the 4 gradients for the sampled windows
        gradients = sample_mean.select(1).subtract(sample_mean.select(7)).abs();
        gradients = gradients.addBands(sample_mean.select(6).subtract(sample_mean.select(2)).abs());
        gradients = gradients.addBands(sample_mean.select(3).subtract(sample_mean.select(5)).abs());
        gradients = gradients.addBands(sample_mean.select(0).subtract(sample_mean.select(8)).abs());

        # And find the maximum gradient amongst gradient bands
        max_gradient = gradients.reduce(ee.Reducer.max());

        # Create a mask for band pixels that are the maximum gradient
        gradmask = gradients.eq(max_gradient);

        # duplicate gradmask bands: each gradient represents 2 directions
        gradmask = gradmask.addBands(gradmask);

        # Determine the 8 directions
        directions = sample_mean.select(1).subtract(sample_mean.select(4)).gt(
            sample_mean.select(4).subtract(sample_mean.select(7))).multiply(1);
        directions = directions.addBands(sample_mean.select(6).subtract(sample_mean.select(4)).gt(
            sample_mean.select(4).subtract(sample_mean.select(2))).multiply(2));
        directions = directions.addBands(sample_mean.select(3).subtract(sample_mean.select(4)).gt(
            sample_mean.select(4).subtract(sample_mean.select(5))).multiply(3));
        directions = directions.addBands(sample_mean.select(0).subtract(sample_mean.select(4)).gt(
            sample_mean.select(4).subtract(sample_mean.select(8))).multiply(4));
        # The next 4 are the not() of the previous 4
        directions = directions.addBands(directions.select(0).Not().multiply(5));
        directions = directions.addBands(directions.select(1).Not().multiply(6));
        directions = directions.addBands(directions.select(2).Not().multiply(7));
        directions = directions.addBands(directions.select(3).Not().multiply(8));

        # Mask all values that are not 1-8
        directions = directions.updateMask(gradmask);

        # "collapse" the stack into a singe band image (due to masking, each pixel has just one value (1-8) in it's directional band, and is otherwise masked)
        directions = directions.reduce(ee.Reducer.sum());

        sample_stats = sample_var.divide(sample_mean.multiply(sample_mean));

        # Calculate localNoiseVariance
        sigmaV = sample_stats.toArray().arraySort().arraySlice(0, 0, 5).arrayReduce(ee.Reducer.mean(), [0]);

        # Set up the 7*7 kernels for directional statistics
        rect_weights = ee.List.repeat(ee.List.repeat(0, 7), 3).cat(ee.List.repeat(ee.List.repeat(1, 7), 4));

        diag_weights = ee.List(
            [[1, 0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0], [1, 1, 1, 0, 0, 0, 0], [1, 1, 1, 1, 0, 0, 0],
             [1, 1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 1, 1, 1]]);

        rect_kernel = ee.Kernel.fixed(7, 7, rect_weights, 3, 3, False);
        diag_kernel = ee.Kernel.fixed(7, 7, diag_weights, 3, 3, False);

        # Create stacks for mean and variance using the original kernels. Mask with relevant direction.
        dir_mean = img.reduceNeighborhood(ee.Reducer.mean(), rect_kernel).updateMask(directions.eq(1));
        dir_var = img.reduceNeighborhood(ee.Reducer.variance(), rect_kernel).updateMask(directions.eq(1));

        dir_mean = dir_mean.addBands(
            img.reduceNeighborhood(ee.Reducer.mean(), diag_kernel).updateMask(directions.eq(2)));
        dir_var = dir_var.addBands(
            img.reduceNeighborhood(ee.Reducer.variance(), diag_kernel).updateMask(directions.eq(2)));

        # and add the bands for rotated kernels
        for i in range(1, 4):
            dir_mean = dir_mean.addBands(
                img.reduceNeighborhood(ee.Reducer.mean(), rect_kernel.rotate(i)).updateMask(directions.eq(2 * i + 1)))
            dir_var = dir_var.addBands(img.reduceNeighborhood(ee.Reducer.variance(), rect_kernel.rotate(i)).updateMask(
                directions.eq(2 * i + 1)))
            dir_mean = dir_mean.addBands(
                img.reduceNeighborhood(ee.Reducer.mean(), diag_kernel.rotate(i)).updateMask(directions.eq(2 * i + 2)))
            dir_var = dir_var.addBands(img.reduceNeighborhood(ee.Reducer.variance(), diag_kernel.rotate(i)).updateMask(
                directions.eq(2 * i + 2)))

        # "collapse" the stack into a single band image (due to masking, each pixel has just one value in it's directional band, and is otherwise masked)
        dir_mean = dir_mean.reduce(ee.Reducer.sum());
        dir_var = dir_var.reduce(ee.Reducer.sum());

        # A finally generate the filtered value
        varX = dir_var.subtract(dir_mean.multiply(dir_mean).multiply(sigmaV)).divide(sigmaV.add(1.0))

        b = varX.divide(dir_var)
        result = dir_mean.add(b.multiply(img.subtract(dir_mean)))

        return result.arrayProject([0]).arrayFlatten([['sum']]).float()

    result = ee.ImageCollection(bandNames.map(inner)).toBands().rename(bandNames).copyProperties(image)

    return image.addBands(result, None, True)

def leesigma_wrap(image, KERNEL_SIZE=30):
    # parameters
    Tk = ee.Image.constant(7)  # number of bright pixels in a 3x3 window
    sigma = 0.9
    enl = 4
    target_kernel = 3
    bandNames = image.bandNames().remove('angle')
    # compute the 98 percentile intensity
    z98 = ee.Dictionary(image.select(bandNames).reduceRegion(
        reducer=ee.Reducer.percentile([98]),
        geometry=image.geometry(),
        scale=10,
        maxPixels=1e13
    )).toImage()
    # select the strong scatterers to retain
    brightPixel = image.select(bandNames).gte(z98)
    K = brightPixel.reduceNeighborhood(ee.Reducer.countDistinctNonNull()
                                       , ee.Kernel.square(target_kernel / 2))
    retainPixel = K.gte(Tk)

    # compute the a-priori mean within a 3x3 local window
    # original noise standard deviation since the data is 5 look
    eta = 1.0 / math.sqrt(enl)
    eta = ee.Image.constant(eta)
    # MMSE applied to estimate the apriori mean
    reducers = ee.Reducer.mean().combine( \
        reducer2=ee.Reducer.variance(), \
        sharedInputs=True
    )
    stats = image.select(bandNames).reduceNeighborhood( \
        reducer=reducers, \
        kernel=ee.Kernel.square(target_kernel / 2, 'pixels'), \
        optimization='window')
    meanBand = bandNames.map(lambda bandName: ee.String(bandName).cat('_mean'))
    varBand = bandNames.map(lambda bandName: ee.String(bandName).cat('_variance'))

    z_bar = stats.select(meanBand)
    varz = stats.select(varBand)

    oneImg = ee.Image.constant(1)
    varx = (varz.subtract(z_bar.abs().pow(2).multiply(eta.pow(2)))).divide(oneImg.add(eta.pow(2)))
    b = varx.divide(varz)
    xTilde = oneImg.subtract(b).multiply(z_bar.abs()).add(b.multiply(image.select(bandNames)))

    # step 3: compute the sigma range
    # Lookup table (J.S.Lee et al 2009) for range and eta values for intensity (only 4 look is shown here)
    LUT = ee.Dictionary({0.5: ee.Dictionary({'I1': 0.694, 'I2': 1.385, 'eta': 0.1921}),
                         0.6: ee.Dictionary({'I1': 0.630, 'I2': 1.495, 'eta': 0.2348}),
                         0.7: ee.Dictionary({'I1': 0.560, 'I2': 1.627, 'eta': 0.2825}),
                         0.8: ee.Dictionary({'I1': 0.480, 'I2': 1.804, 'eta': 0.3354}),
                         0.9: ee.Dictionary({'I1': 0.378, 'I2': 2.094, 'eta': 0.3991}),
                         0.95: ee.Dictionary({'I1': 0.302, 'I2': 2.360, 'eta': 0.4391})});

    # extract data from lookup
    sigmaImage = ee.Dictionary(LUT.get(str(sigma))).toImage()
    I1 = sigmaImage.select('I1')
    I2 = sigmaImage.select('I2')
    # new speckle sigma
    nEta = sigmaImage.select('eta')
    # establish the sigma ranges
    I1 = I1.multiply(xTilde)
    I2 = I2.multiply(xTilde)

    # step 3: apply MMSE filter for pixels in the sigma range
    # MMSE estimator
    mask = image.select(bandNames).gte(I1).Or(image.select(bandNames).lte(I2))
    z = image.select(bandNames).updateMask(mask)

    stats = z.reduceNeighborhood( \
        reducer=reducers, \
        kernel=ee.Kernel.square(KERNEL_SIZE / 2, 'pixels'), \
        optimization='window')

    z_bar = stats.select(meanBand)
    varz = stats.select(varBand)

    varx = (varz.subtract(z_bar.abs().pow(2).multiply(nEta.pow(2)))).divide(oneImg.add(nEta.pow(2)))
    b = varx.divide(varz)
    # if b is negative set it to zero
    new_b = b.where(b.lt(0), 0)
    xHat = oneImg.subtract(new_b).multiply(z_bar.abs()).add(new_b.multiply(z))

    # remove the applied masks and merge the retained pixels and the filtered pixels
    xHat = image.select(bandNames).updateMask(retainPixel).unmask(xHat)
    output = ee.Image(xHat).rename(bandNames)
    return image.addBands(output, None, True)

def leesigma(KERNEL_SIZE):
    '''
    KERNEL_SIZE : 越小计算时间越长
    '''
    return partial(leesigma_wrap, KERNEL_SIZE=KERNEL_SIZE)
