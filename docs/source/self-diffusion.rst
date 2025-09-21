
Self-diffusion coefficient
==========================

``vacf.py`` calculates self-diffusion coefficient from particle velocities. The self-diffusion coefficient,
:math:`D`, is computed from an exponential fit to the running integral of velocity auto-correlation function
(VACF) using the following Green-Kubo relation

.. math::
    D = \frac{1}{3} \int_0^\infty \left\langle \boldsymbol{v}_i(t) \cdot \boldsymbol{v}_i(t_0) \right\rangle dt

where :math:`\boldsymbol{v}_i(t)` denotes the velocity of particle :math:`i` at any specific time :math:`t`.

Usage:

.. code-block:: bash

    python vacf.py -h

.. tip::

    An example data file, velocity.data, is available in the example directory.
    The required values to run the example can be found in the md.param file.


.. autofunction:: vacf.acf

.. autofunction:: vacf.diffusion