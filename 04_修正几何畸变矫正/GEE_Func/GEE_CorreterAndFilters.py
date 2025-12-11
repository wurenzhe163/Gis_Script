import ee
import numpy as np
import math
from functools import partial


class ImageFilter(object):
    # -----------------------------------------------SAR图像滤波-----------------------------------------
    @staticmethod
    def boxcar(KERNEL_SIZE):
        def wrap(image):
            bandNames = image.bandNames().remove('angle')
            # Define a boxcar kernel
            kernel = ee.Kernel.square((KERNEL_SIZE / 2), units='pixels', normalize=True)
            # Apply boxcar
            output = image.select(bandNames).convolve(kernel).rename(bandNames)
            return image.addBands(output, None, True)

        return wrap

    @staticmethod
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

    @staticmethod
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

    @staticmethod
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
                    img.reduceNeighborhood(ee.Reducer.mean(), rect_kernel.rotate(i)).updateMask(
                        directions.eq(2 * i + 1)))
                dir_var = dir_var.addBands(
                    img.reduceNeighborhood(ee.Reducer.variance(), rect_kernel.rotate(i)).updateMask(
                        directions.eq(2 * i + 1)))
                dir_mean = dir_mean.addBands(
                    img.reduceNeighborhood(ee.Reducer.mean(), diag_kernel.rotate(i)).updateMask(
                        directions.eq(2 * i + 2)))
                dir_var = dir_var.addBands(
                    img.reduceNeighborhood(ee.Reducer.variance(), diag_kernel.rotate(i)).updateMask(
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

    @staticmethod
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

    @staticmethod
    def leesigma(KERNEL_SIZE):
        '''
        KERNEL_SIZE : 越小计算时间越长
        '''
        return partial(ImageFilter.leesigma_wrap, KERNEL_SIZE=KERNEL_SIZE)
    
    #-----------------------------------适用于所有图像
    @staticmethod
    def Open_close(img, radius=10):
        '''
        开闭运算
        '''
        uniformKernel = ee.Kernel.square(**{'radius': radius, 'units': 'meters'})
        min = img.reduceNeighborhood(**{'reducer': ee.Reducer.min(), 'kernel': uniformKernel})
        Openning = min.reduceNeighborhood(**{'reducer': ee.Reducer.max(), 'kernel': uniformKernel})
        max = Openning.reduceNeighborhood(**{'reducer': ee.Reducer.max(), 'kernel': uniformKernel})
        Closing = max.reduceNeighborhood(**{'reducer': ee.Reducer.min(), 'kernel': uniformKernel})
        return Closing
    

class S1Corrector(object):
    def volumetric_model_SCF(theta_iRad, alpha_rRad):
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

    def surface_model_SCF(theta_iRad, alpha_rRad, alpha_azRad):
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

    def volumetric(model, theta_iRad, alpha_rRad, alpha_azRad, gamma0):
        '''辐射斜率校正'''
        if model == 'volume':
            scf = S1Corrector.volumetric_model_SCF(theta_iRad, alpha_rRad)
        if model == 'surface':
            scf = S1Corrector.surface_model_SCF(theta_iRad, alpha_rRad, alpha_azRad)

        gamma0_flat = gamma0.divide(scf)
        gamma0_flatDB = (ee.Image.constant(10).multiply(gamma0_flat.log10()).select(['VV', 'VH'], ['VV_gamma0flat',
                                                                                                   'VH_gamma0flat']))
        ratio_flat = (gamma0_flatDB.select('VV_gamma0flat').subtract(
            gamma0_flatDB.select('VH_gamma0flat')).rename('ratio_gamma0flat'))

        return {'scf': scf, 'gamma0_flat': gamma0_flat,
                'gamma0_flatDB': gamma0_flatDB, 'ratio_flat': ratio_flat}

    def getS1Corners(image, AOI_buffer, orbitProperties_pass):
        '''
        azimuthEdge ： S1的方位角模拟线，{'azimuth': azimuth}对应方位角
        rotationFromNorth ：S1距离角
        startpoint ： Ascending左上角 或 Decending右上角
        endpoint ： Ascending右下角 或 Decending左下角
        coordinates_dict ：  S1的四个角点坐标，以及经纬度
        '''
        # 真实方位角(根据整幅影响运算)
        coords = ee.Array(image.geometry().coordinates().get(0)).transpose()
        crdLons = ee.List(coords.toList().get(0))
        crdLats = ee.List(coords.toList().get(1))
        minLon = crdLons.sort().get(0)
        maxLon = crdLons.sort().get(-1)
        minLat = crdLats.sort().get(0)
        maxLat = crdLats.sort().get(-1)
        azimuth = (ee.Number(crdLons.get(crdLats.indexOf(minLat))).subtract(minLon).atan2(
            ee.Number(crdLats.get(crdLons.indexOf(minLon))).subtract(minLat))
                   .multiply(180.0 / math.pi))

        if orbitProperties_pass == 'ASCENDING':
            azimuth = azimuth.add(270.0)
            rotationFromNorth = azimuth.subtract(360.0)
        elif orbitProperties_pass == 'DESCENDING':
            azimuth = azimuth.add(180.0)
            rotationFromNorth = azimuth.subtract(180.0)
        else:
            raise TypeError

        # S1方位角，构筑一条线{'azimuth': azimuth}对应角度
        azimuthEdge = (ee.Feature(ee.Geometry.LineString([crdLons.get(crdLats.indexOf(minLat)),
                                                          minLat, minLon,
                                                          crdLats.get(crdLons.indexOf(minLon))]),
                                  {'azimuth': azimuth}).copyProperties(image))
        
        # 关于Buffer计算辅助线(根据局部范围运算)
        coords = ee.Array(image.clip(AOI_buffer).geometry().coordinates().get(0)).transpose()
        crdLons = ee.List(coords.toList().get(0))
        crdLats = ee.List(coords.toList().get(1))
        minLon = crdLons.sort().get(0)
        maxLon = crdLons.sort().get(-1)
        minLat = crdLats.sort().get(0)
        maxLat = crdLats.sort().get(-1)

        if orbitProperties_pass == 'ASCENDING':
            # 左上角
            startpoint = ee.List([minLon, maxLat])
            # 右下角
            endpoint = ee.List([maxLon, minLat])
        elif orbitProperties_pass == 'DESCENDING':
            # 右上角
            startpoint = ee.List([maxLon, maxLat])
            # 左下角
            endpoint = ee.List([minLon, minLat])

        coordinates_dict = {'crdLons': crdLons, 'crdLats': crdLats,
                            'minLon': minLon, 'maxLon': maxLon, 'minLat': minLat, 'maxLat': maxLat}

        return azimuthEdge, rotationFromNorth, startpoint, endpoint, coordinates_dict


