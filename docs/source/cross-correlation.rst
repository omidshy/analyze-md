
Velocity cross-correlation function
===================================

The collective dynamics of different particles in a MD simulation can be investigated
through velocity cross-correlation functions (VCCF).
``vccf.py`` computes the time cross-correlation function between the initial velocity of a
particle of type :math:`i` and the latter velocities of particles of type :math:`j`,
initially located inside a spherical shell of radius :math:`R` around the particle of type :math:`i`,
defined as

.. math::
    C(t) = \frac{ \left\langle \boldsymbol{v}_j(t) \cdot \boldsymbol{v}_i(0) \right\rangle }
    { \left( \left\langle v_i^2 \right\rangle \left\langle v_j^2 \right\rangle \right) ^{0.5} }

where :math:`\boldsymbol{v}_i(t)` and :math:`\boldsymbol{v}_j(t)` are the velocity of the particles :math:`i`
and :math:`j` at any specific time, :math:`\left\langle v_i^2 \right\rangle` and :math:`\left\langle v_j^2 \right\rangle`
are the mean squared velocities of all particles of type :math:`i` and :math:`j`, respectively.
:math:`\left\langle \boldsymbol{v}_j(t) \cdot \boldsymbol{v}_i(0) \right\rangle` is a restricted
statistical average defined as

.. math::
    \frac{1}{N} \left\langle \sum_{j} \boldsymbol{v}_j(t) \cdot \boldsymbol{v}_i(0) \cdot u( R - r_{ij}(0) ) \right\rangle

where :math:`u(x)` is the step function, :math:`r_{ij}(0)` is the initial distance between the central
particle :math:`i` and a particle :math:`j`, and :math:`N` is the mean number of :math:`j`
particles in the spherical shell of a particle :math:`i`.
The cutoff radius :math:`R` for species :math:`i` and :math:`j` is usually set to the position of the first minimum of
their center-of-mass radial distribution function.

Usage:

.. code-block:: bash

    python vccf.py -h

.. tip::

    Example data files, particle_a.data and particle_b.data, are available in the example directory.
    The required values to run the example can be found in the md.param file.


.. autofunction:: vccf.ccf