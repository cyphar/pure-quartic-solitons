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
# TODO: What is \Gamma?
GAMMA = 1

# First of all, we're interested in integrating from the wings. If you
# look at the key equation from the paper you can ignore the
# non-linearity when the amplitude is small (|A| ~ 0). This results in a
# simple fourth-order linear ODE:
#
#        \frac{dA}{dz} = i \frac{\beta_4}{24} \frac{d^4 a}{dt^4}
#
# Which then the solutions are given by:
# XXX
#
# We can then reduce the problem space from 8 real parameters (4 complex
# parameters) by observing
def init(a, b, c):
	return [random.random() * 0.001 + random.random() * 0.001j] + [0, 0, 0]

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
	start = init(0, 0, 0)

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
	t0 = 0.0
	t1 = 20.0
	dt = 0.001

	out = "test-%d.png" % (random.randrange(0, 999999),)

	fig = plt.figure(figsize=(10, 10), dpi=80)
	ax = latexify(fig.add_subplot("111"))

	for i in range(5):
		ts, As = solve(t0, t1, dt)
		ax.plot(ts, abs(As) ** 2)

	ax.set_axisbelow(True)
	ax.set_xlabel("Time (s)")
	ax.set_ylabel("Intensity ($|A|^2$)")
	ax.set_xlim([t0, t1])
	ax.set_ylim([0, 15])

	plt.legend()
	fig.tight_layout()

	plt.savefig(out)
	print(out)


if __name__ == "__main__":
	main()
