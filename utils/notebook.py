# Returns True if the current execution context is an IPython notebook, e.g. Jupyter.
# https://stackoverflow.com/questions/15411967/how-can-i-check-if-code-is-executed-in-the-ipython-notebook

# Poor man's cache
IN_IPYNB = None

def in_ipynb():
  if IN_IPYNB is not None:
    return IN_IPYNB

  try:
    cfg = get_ipython().config
    if str(type(get_ipython())) == "<class 'ipykernel.zmqshell.ZMQInteractiveShell'>":
    # if cfg['IPKernelApp']['parent_appname'] == 'ipython-notebook':
      # print ('Running in ipython notebook env.')
      return True
    else:
      return False
  except NameError:
    # print ('NOT Running in ipython notebook env.')
    return False
