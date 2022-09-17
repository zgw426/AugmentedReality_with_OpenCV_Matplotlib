# Augmented Reality (AR) with OpenCV & Matplotlib

Map graphs drawn with Matplotlib to AR markers with OpenCV.


## 環境

It is Python3.9 in Windows11 environment.
I have a camera device connected to my Windows computer.

```console:CMD
C:\>ver

Microsoft Windows [Version 10.0.22000.978]
```

```console:PowerShell
PS C:\> python --version
Python 3.9.5
```

## set up

Install the required modules with the following command.

```console
pip install -r requirements.txt
```

## execution

### Step01.Mapping a 2D graph in Matplotlib to an input image

Map the 2D graph of Matplotlib to the AR marker of the input image (sample.png) and output the result as the output image (out-sample.png).

execution command

```console
python 01_matplot2D_NotCamera.py
```

### Step02.Mapping Matplotlib 2D graphs to AR markers in camera footage

Map 2D graphs in Matplotlib to AR markers in camera footage.

[![](https://img.youtube.com/vi/t4DyeLGA0gk/0.jpg)](https://www.youtube.com/watch?v=t4DyeLGA0gk)

execution command

```console
python 02_matplot2d_no-transparency.py
```

Quit with the q key.

### Step03.Mapping Matplotlib 3D graph to AR markers in camera footage

Map 3D graphs in Matplotlib to AR markers in camera footage.

[![](https://img.youtube.com/vi/vzQA8_DQ8tw/0.jpg)](https://www.youtube.com/watch?v=vzQA8_DQ8tw)

execution command

```console
python 03_matplot3d_no-transparency.py
```

Quit with the q key.

### Step04.Mapping a 3D graph in Matplotlib with transparency processing on AR markers in camera video

Map 3D graphs in Matplotlib to AR markers in camera footage.
Make the background of the 3D graph transparent with transparency processing.

[![](https://img.youtube.com/vi/QnYwcrHeKbI/0.jpg)](https://www.youtube.com/watch?v=QnYwcrHeKbI)

execution command

```console
python 04_matplot3d_transparency.py
```

Quit with the q key.


## digression

Docker is used as the execution environment.
However, only 01_matplot2D_NotCamera.py works because the camera device cannot be recognized in Windows environment.
(There seems to be a way to recognize camera devices, but I stopped investigating because it was troublesome.)

### Build a Docker image

```console
docker build -f Dockerfile -t pyimg .
```

### Start Docker container

```console
docker run --rm --name python -v ${PWD}:/work -dit pyimg
```

### Enter Docker container

```console
docker exec -it python bash
```

### Run 01_matplot2D_NotCamera.py

Execute `Step 01`.

execution command

```console
python 01_matplot2D_NotCamera.py
```

The output image (out-sample.png) is the result of mapping a Matplotlib 2D graph to the input image (sample.png).

# Stop Docker container

```console
docker stop python
```
