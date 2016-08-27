#!/usr/bin/env python3
# Copyright (C) 2016 Aleksa Sarai
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import cmath
import matplotlib.pyplot as plt
import numpy
import random
import scipy
import scipy.integrate

SPINE_COLOR = "black"

def latexify(ax):
	for spine in ["top", "right"]:
		ax.spines[spine].set_visible(False)

	for spine in ["left", "bottom"]:
		ax.spines[spine].set_color(SPINE_COLOR)
		ax.spines[spine].set_linewidth(0.5)

	ax.xaxis.set_ticks_position("bottom")
	ax.yaxis.set_ticks_position("left")

	for axis in [ax.xaxis, ax.yaxis]:
		axis.set_tick_params(direction="out", color=SPINE_COLOR)

	return ax

# The key equation (from the paper):
# \frac{\partial A}{\partial z} =
#         i \frac{\beta_4}{24} \frac{\partial^4 A}{\partial t^4} +
#         i \gamma_{eff} {\left|A\right|}^{2} A

# This is a configuration parameter for the fourth-order ODE.
GAMMA = 1

# First of all, we're interested in integrating from the wings. If you
# look at the key equation from the paper you can ignore the
# non-linearity when the amplitude is small (|A| ~ 0). This results in a
# simple fourth-order linear ODE:
#
# \frac{\partial A}{\partial z} =
#         i \frac{\beta_4}{24} \frac{\partial^4 a}{\partial t^4}
#
# You apply the standard assumption, that A is of the form:
#
#                       A(z,t) = e^{i\Gamma t}a(t)
#
# This reduces the PDE to an ODE:
#
#             \Gamma a(t) = \frac{\beta_4}{24} \frac{d^4 a}{dt^4}
#
# Which (knowing that \beta_4 < 0) gives you the following roots of the
# characteristic equation:
#
#         \lambda_N = \frac{24}{\beta_4} \Gamma e^{i N\frac{pi}{4}
#                   = \frac{24}{\beta_4} \Gamma \left(\pm 1 \pm i\right)
#                  ~= \Gamma e^{i N\frac{pi}{4}}
#
# Which then becomes the general solution:
#           a = \alpha e^{\lambda_1 t} + \beta  e^{\lambda_2 t}
#               \gamma e^{\lambda_3 t} + \delta e^{\lambda_4 t}
#
# We can then reduce the problem space from 8 real parameters (4 complex
# parameters) by observing that in either side of the wings, only two of the
# terms are significant at large (or small) t (because the terms are either
# weighted by e^t or e^{-t}). So, you can now deal with just 2 complex
# parameters. In addition, solutions of an ODE can be rotated in phase space
# and will still be solutions. Thus, by picking an arbitrary \theta, you can
# reduce the problem to only 3 real parameters (1 complex, 1 real).
def init(a, b, c, t):
	lamb1 = GAMMA * cmath.exp(1j * cmath.pi/4.0 + 0 * cmath.pi/2.0)
	lamb2 = GAMMA * cmath.exp(1j * cmath.pi/4.0 + 1 * cmath.pi/2.0)

	return [           a * cmath.exp(lamb1 * t) +            (a + b*1j) * cmath.exp(lamb2 * t),
	        lamb1    * a * cmath.exp(lamb1 * t) + lamb2    * (a + b*1j) * cmath.exp(lamb2 * t),
	        lamb1**2 * a * cmath.exp(lamb1 * t) + lamb2**2 * (a + b*1j) * cmath.exp(lamb2 * t),
	        lamb1**3 * a * cmath.exp(lamb1 * t) + lamb2**3 * (a + b*1j) * cmath.exp(lamb2 * t)]

# We need to first convert the key equation from the paper to an ODE. If
# you assume that A is of the form:
#
#                       A(z,t) = e^{i\Gamma t}a(t)
#
# This is the end result:
#
#   \frac{d^4 a}{dt^4} - \Gamma a(t) + {\left|a(t)\right|}^2 a(t) = 0
#
# We then can reduce this to the following set of first-order ODEs,
# which Python can actually solve:
#
#        \alpha' &= \beta
#        \beta'  &= \gamma
#        \gamma' &= \Delta
#        \Delta' &= \Gamma\alpha - {\left|\alpha\right|}^2 \alpha
def soliton(t, y):
	a, b, c, d = y
	return [b, c, d, GAMMA*a - a*abs(a)**2]

# Solver.
def solve(t0, t1, dt):
	a = random.random() * 0.001
	b = random.random() * 0.001
	c = random.random() * 0.001
	start = init(a, b, c, t0)

	solver = scipy.integrate.ode(soliton)
	solver.set_integrator("zvode", method="adams", order=10)
	solver.set_initial_value(start, t0)

	As = []
	while solver.successful() and solver.t < t1:
		solver.integrate(solver.t + dt)
		As.append((solver.t, solver.y[0]))
		print("%s %s" % (solver.t, solver.y))

	ts, As = numpy.array(As).T

	return ts, As

def main():
	t0 = 1e-10
	t1 = 30
	dt = 0.001

	out = "test-%d.png" % (random.randrange(0, 999999),)

	fig = plt.figure(figsize=(10, 10), dpi=80)
	ax1 = latexify(fig.add_subplot("211"))
	ax2 = latexify(fig.add_subplot("212"))

	for i in range(1):
		# TODO: Plot the amplitude and phase.
		#  XXX: Is the phase going to be flat? Is that a property of the DEs?
		ts, As = solve(t0, t1, dt)
		ax1.plot(ts, abs(As))
		ax2.plot(ts, cmath.phase(As))

	ax.set_axisbelow(True)
	ax.set_xlabel("Time (s)")
	ax.set_ylabel("Intensity ($|A|^2$)")
	ax.set_xlim([t0, t1])
	ax.set_ylim([0, None])

	plt.legend()
	fig.tight_layout()

	plt.savefig(out)
	print(out)


if __name__ == "__main__":
	main()
