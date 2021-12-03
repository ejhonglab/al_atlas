
This repository contains data from:

Grabe V, Baschwitz A, Dweck HKM, Lavista-Llanos S, Hansson BS, Sachse S. Elucidating the Neuronal Architecture of Olfactory Glomeruli in the Drosophila Antennal Lobe. Cell Rep. 2016 Sep 20;16(12):3401-3413. doi: [10.1016/j.celrep.2016.08.063](https://doi.org/10.1016/j.celrep.2016.08.063)


### Previously online data

The following files were previously available here:

http://www.ice.mpg.de/ext/index.php?id=invivoALatlas

- [invivoALatlas.pdf](invivoALatlas.pdf?raw=1)
- [invivoALstack.tif](invivoALstack.tif?raw=1)

...though now that link is dead and I could not find another link after a bit of
searching.

The same dead link may have also had something like a `.lsm` file that either contained
glomerulus segmentation information (perhaps equivalent/similar to some of the Amira
formats under `from_veit/`).

I was able to interact with the 3D PDF data using Adobe Acrobat 9.5.5 on Ubuntu 20.04.
Current versions of Adobe Acrobat may also work, but are just no longer supported on
Linux. The stock Ubuntu PDF reader did not render the 3D data.

The PDF format has the 3D segmentation information to qualitatively interact with, but I
found it not practically convertable into more useful 3D formats.


#### Getting Adobe Acrobat working on Linux

Instructions (for Ubuntu 20.04) copied from [here](https://linuxconfig.org/how-to-install-adobe-acrobat-reader-on-ubuntu-20-04-focal-fossa-linux):
```
wget -O ~/adobe.deb ftp://ftp.adobe.com/pub/adobe/reader/unix/9.x/9.5.5/enu/AdbeRdr9.5.5-1_i386linux_enu.deb

sudo dpkg --add-architecture i386
sudo apt update

sudo apt install libxml2:i386 libcanberra-gtk-module:i386 gtk2-engines-murrine:i386 libatk-adaptor:i386

sudo dpkg -i ~/adobe.deb
```

Then you can open the PDF via:
```
acroread invivoALatlas.pdf
```


### Additional exports

The folder `from_veit/` contain exports of the same data as should be in the PDF /
whatever format used to be on their website, into other formats I tried. Veit Grabe and
Silke Sasche kindly helped me get these alternative formats.

The `.am` file can seemingly be loaded into Fiji with the `Import -> Amira...` option,
though it's unclear how to get this into a more useful format. It also seems as though
the file might not be fully-correctly-loaded by Fiji, as there is a console error that
appears while loading it.

The same import option seemed to stall using the `.surf.am` file.

The scripts `make_atlas_from_[am/obj].py` were my attempts to get the specific glomeruli
out as individual volumes using some of these other export formats.

