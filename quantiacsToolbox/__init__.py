import pkg_resources

try:
    pkg_resources.get_distribution("quantiacstoolbox")
except DistributionNotFound:
    from .quantiacsToolbox import runts, loadData, plotts, stats, submit, computeFees, updateCheck
