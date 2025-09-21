
Viscosity
=========

``visco.py`` calculates viscosity using components of the pressure tensor obtained from a canonical ensemble (NVT)
molecular dynamics (MD) simulation. Employs the Einstein or Green-Kubo relation, where viscosity is determined from
the integral of the pressure tensor elements or their auto-correlation function, respectively.

The viscosity, :math:`\eta`, is calculated from the integral of the pressure tensor auto-correlation
function over time following the Green--Kubo approach

.. math::
    \eta = \frac{V}{k_B T} \int_0^\infty \left\langle P_{\alpha \beta} \left( t \right)
    \cdot P_{\alpha \beta} \left( t_0 \right) \right\rangle dt ,

or the Einstein approach

.. math::
    \eta = \lim_{t \to \infty} \frac{V}{2 t k_B T}
    \left\langle \left( \int_0^\infty P_{\alpha \beta}(t') dt' \right)^2  \right\rangle

where :math:`V` is the simulation box volume, :math:`k_B` is the Boltzmann constant, :math:`T` is temperature,
:math:`P_{\alpha \beta}` denotes the off-diagonal element :math:`\alpha \beta` of the pressure tensor,
and the brackets indicate that average must be taken over all time origins :math:`t_0`.

Usage:

.. code-block:: bash

    python visco.py -h

.. tip::

    An example data file, press.data, is available in the example directory.
    The required values to run the example can be found in the md.param file.


.. autofunction:: visco.einstein

.. autofunction:: visco.green_kubo