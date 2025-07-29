
# Python codes for MD simulation analysis

A collection of Python scripts for computing physical properties and analyzing trajectories from
molecular dynamics (MD) simulations.

## visco.py

Calculates viscosity using components of the pressure tensor obtained from a canonical ensemble (NVT)
molecular dynamics (MD) simulation. Employs the Einstein or Green-Kubo relation, where viscosity is
determined from the integral of the pressure tensor elements or their auto-correlation function, respectively.

The viscosity, *η*, is calculated from the integral of the pressure tensor auto-correlation
function over time following the Green--Kubo approach

<!--<picture>
  <source
    media="(prefers-color-scheme: dark)"
    srcset="assets/visco_gk_dark.png"
  />
  <source
    media="(prefers-color-scheme: light)"
    srcset="assets/visco_gk_light.png"
  />
  <img
    alt="viscosity Green--Kubo equation"
    src="assets/visco_gk_light.png"
    height="45"
  />
</picture>-->

$$
\eta = \frac{V}{k_B T} \int_0^\infty \left\langle P_{\alpha \beta} \left( t \right)
\cdot P_{\alpha \beta} \left( t_0 \right) \right\rangle dt
$$

or the Einstein approach

<!--<picture>
  <source
    media="(prefers-color-scheme: dark)"
    srcset="assets/visco_en_dark.png"
  />
  <source
    media="(prefers-color-scheme: light)"
    srcset="assets/visco_en_light.png"
  />
  <img
    alt="viscosity Einstein equation"
    src="assets/visco_en_light.png"
    height="58"
  />
</picture>-->

$$
\eta = \lim_{t \to \infty} \frac{V}{2 t k_B T}
\left\langle \left( \int_0^\infty P_{\alpha \beta}(t') dt' \right)^2  \right\rangle
$$

where *V* is the simulation box volume, *k<sub>B</sub>* is the Boltzmann constant, *T* is temperature,
*P<sub>αβ</sub>* denotes the off-diagonal element *αβ* of the pressure tensor,
and the brackets indicate that average must be taken over all time origins *t<sub>0</sub>*.

Usage: `python visco.py -h`

> [!TIP]
> An example data file, press.data, is available in the example directory.
> The required values to run the example can be found in the md.param file.

## vacf.py

Calculates self-diffusion coefficient from particle velocities. The self-diffusion coefficient, *D*, is
computed from an exponential fit to the running integral of velocity auto-correlation function (VACF)
using the following Green-Kubo relation

<!--<picture>
  <source
    media="(prefers-color-scheme: dark)"
    srcset="assets/sdc_gk_dark.png"
  />
  <source
    media="(prefers-color-scheme: light)"
    srcset="assets/sdc_gk_light.png"
  />
  <img
    alt="diffusion Green--Kubo equation"
    src="assets/sdc_gk_light.png"
    height="45"
  />
</picture>-->

$$
D = \frac{1}{3} \int_0^\infty \left\langle \mathbf{v}_i(t) \cdot \mathbf{v}_i(t_0) \right\rangle dt
$$

where ***v**<sub>i</sub>(t)* denotes the velocity of particle *i* at any specific time *t*.

Usage: `python vacf.py -h`

> [!TIP]
> An example data file, velocity.data, is available in the example directory.
> The required values to run the example can be found in the md.param file.

## vccf.py

The collective dynamics of different particles in a MD simulation can be investigated
through velocity cross-correlation functions.
The time cross-correlation function between the initial velocity of a particle of type *i* and
the latter velocities of particles of type *j*, initially located inside a spherical shell of
radius *R* around the particle of type *i*, can be defined as

$$
C(t) = \frac{ \left\langle \mathbf{v}_j(t) \cdot \mathbf{v}_i(0) \right\rangle }
{ \left( \left\langle v_i^2 \right\rangle \left\langle v_j^2 \right\rangle \right) ^{0.5} }
$$

where $\mathbf{v}_i(t)$ and $\mathbf{v}_j(t)$ are the velocity of the particles $i$
and $j$ at any specific time, $\left\langle v_i^2 \right\rangle$ and $\left\langle v_j^2 \right\rangle$
are the mean squared velocities of all particles of type $i$ and $j$, respectively.
$\left\langle \mathbf{v}_j(t) \cdot \mathbf{v}_i(0) \right\rangle$ is a restricted
statistical average defined as

$\frac{1}{N} \left\langle \sum_{j} \mathbf{v}_j(t) \cdot \mathbf{v}_i(0) \cdot u( R - r_{ij}(0) ) \right\rangle$

where $u(x)$ is the step function, $r_{ij}(0)$ is the initial distance between the central
particle $i$ and a particle $j$, and $N$ is the mean number of $j$
particles in the spherical shell of a particle $i$.
The cutoff radius $R$ for species $i$ and $j$ is usually set to the position of the first minimum of
their center-of-mass radial distribution function.

Usage: `python vccf.py -h`

> [!TIP]
> Example data files, particle_a.data and particle_b.data, are available in the example directory.
> The required values to run the example can be found in the md.param file.
