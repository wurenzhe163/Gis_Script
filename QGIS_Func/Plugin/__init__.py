def classFactory(iface):
    from .merge_tiff import MergeTIFFPlugin
    return MergeTIFFPlugin(iface)