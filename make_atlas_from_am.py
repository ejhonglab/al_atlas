#!/usr/bin/env python3

"""
`ahds` was unable to load the .surf.am file, which seems to have the actual geometry
data. It could load the .am file.

trying the .surf.am file:
```
Using pattern: b'(?:^|\n)@(?P<stream>\\d+)\n'
Traceback (most recent call last):
  File "./al_atlas.py", line 20, in <module>
    main()
  File "./al_atlas.py", line 14, in main
    surf = AmiraFile(prefix + 'surf.am')
  File "/home/tom/src/al_atlas/venv/lib/python3.8/site-packages/ahds/__init__.py", line 66, in __init__
    self._header = AmiraHeader(fn, load_streams=load_streams, *args, **kwargs)
  File "/home/tom/src/al_atlas/venv/lib/python3.8/site-packages/ahds/header.py", line 151, in __init__
    self._load()
  File "/home/tom/src/al_atlas/venv/lib/python3.8/site-packages/ahds/header.py", line 212, in _load
    _parameters = self._load_parameters(block_data['parameters'], 'Parameters', parent=self)
  File "/home/tom/src/al_atlas/venv/lib/python3.8/site-packages/ahds/header.py", line 331, in _load_parameters
    self._load_parameters(param['parameter_value'], name=param['parameter_name']))
  File "/home/tom/src/al_atlas/venv/lib/python3.8/site-packages/ahds/header.py", line 336, in _load_parameters
    block.add_attr(param['parameter_name'], param['parameter_value'])
  File "/home/tom/src/al_atlas/venv/lib/python3.8/site-packages/ahds/core.py", line 148, in add_attr
    raise ValueError("will not overwrite attribute '{}'".format(attr_name))
ValueError: will not overwrite attribute 'name'
```
"""

from os.path import join

from ahds import AmiraFile, AmiraHeader


def main():
    amira_dir = 'cs-transfer'
    prefix = join(amira_dir, 'Merged_2-101221a-labels_only_sure_ones_Sensillarcolors.')

    #am = AmiraFile(prefix + 'am')
    #print(am)

    # all fail (.surf was just symlinked to the .surf.am i got)
    #surf_header = AmiraHeader(prefix + 'surf')
    #print(surf_header)
    #surf = AmiraFile(prefix + 'surf')
    #print(surf)
    #surf = AmiraFile(prefix + 'surf.am')
    #print(surf)

    import ipdb; ipdb.set_trace()


if __name__ == '__main__':
    main()

