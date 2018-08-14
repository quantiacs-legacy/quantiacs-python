import pkg_resources

moduleList = [d for d in pkg_resources.working_set]

if 'quantiacstoolbox' in moduleList:
    from .quantiacsToolbox import runts, loadData, plotts, stats, submit, computeFees, updateCheck
