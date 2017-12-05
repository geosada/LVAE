# LVAE

TensorFlow implementation of [Ladder Variational Autoencoders](https://papers.nips.cc/paper/6275-ladder-variational-autoencoders.pdf) (NIPS 2016) on the MNIST generation task.

<div style="text-align: center;">
  <img src="https://raw.githubusercontent.com/geosada/LVAE/img/reconst.png" width="112">
  <img src="https://raw.githubusercontent.com/geosada/LVAE/img/reconst2.png" width="112">
  <img src="https://raw.githubusercontent.com/geosada/LVAE/img/reconst3.png" width="112">
  <img src="https://raw.githubusercontent.com/geosada/LVAE/img/reconst4.png" width="112">
</div>

Right are original inputs and left are reconstructed images.
This code was written considerably inspired from [Eric Jang's DRAW](https://github.com/ericjang/draw).


## Architecture

<div style="text-align: center;">
  <img src="https://raw.githubusercontent.com/geosada/LVAE/img/LVAE.png" width=100%>
</div>


## Usage

```python main.py```


## Useful Resources

- [Eric Jang's implementaion of DRAW](https://github.com/ericjang/draw)
- [Casper Sønderby's implementaion with Thaeno and Parmesan](https://github.com/casperkaae/LVAE)
