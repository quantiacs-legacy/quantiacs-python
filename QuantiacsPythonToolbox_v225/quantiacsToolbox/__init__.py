import pip
moduleList = [i.key for i in pip.get_installed_distributions()]

if 'quantiacstoolbox' in moduleList:
    from .quantiacsToolbox import runts, loadData, plotts, stats, submit, computeFees, updateCheck
