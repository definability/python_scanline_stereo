# Python Scanline Stereo

Implementation of scanline stereo vision problem solution
using dynamic programming on chain graphs
using Python + [NumPy][numpy] and [Numba][numba].

Check out the following Medium stories to know more about this project:

- [Dynamic Programming on Chain Graphs. Part 1: The Problem][medium-part1];
- [Dynamic Programming on Chain Graphs. Part 2: The Solution][medium-part2];
- [Scanline Stereo Vision][medium-stereo-vision];
- [Efficient Python Implementation of Scanline Stereo Vision][medium-python-stereo].

## Installing Dependencies

You need Python 3.12 and [pipenv].
Then, create a virtual environment and install the needed dependencies:
```bash
pipenv shell
pipenv install
```

Alternatively, you can use [venv] or any other virtual environment tool
and install the following packages:
- [pillow] 10.4.0
- [numpy 2.0.0][numpy-2]
- [numba 0.60.0][numba-0.60]

Keep in mind that [numba 0.60.0][numba-0.60]
is not compatible with [NumPy][numpy] versions higher than 2.0.

## Launching

Use help to know more:
```bash
python . -h
```

If you have two files `view0.png` and `view1.png`
and want to launch the algorithm using maximum disparity `50`
with smoothness term `25` and save the result to `out.png`,
using the NumPy-based implementation, use
```bash
python . -l view0.png -r view1.png -o out.png -d 50 -e 25 -s numpy
```

[medium-part1]: https://medium.com/@valeriy.krygin/dynamic-programming-on-chain-graphs-part-1-the-problem-78bcf0250257
[medium-part2]: https://medium.com/@valeriy.krygin/dynamic-programming-on-chain-graphs-part-2-the-solution-37c1bad8570e
[medium-stereo-vision]: https://medium.com/@valeriy.krygin/scanline-stereo-vision-85ff252ec521
[medium-python-stereo]: https://medium.com/@valeriy.krygin/efficient-python-implementation-of-scanline-stereo-vision-c7dcff677b3c
[numba]: https://numba.pydata.org
[numba-0.60]: https://numba.readthedocs.io/en/stable/release/0.60.0-notes.html
[numpy]: https://numpy.org
[numpy-2]: https://numpy.org/devdocs/release/2.0.0-notes.html
[pillow]: https://python-pillow.org
[pipenv]: https://pipenv.pypa.io/en/latest
[venv]: https://docs.python.org/3/library/venv.html
