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

import math
import cmath
import csv
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid.anchored_artists import AnchoredText
import numpy
import numpy.random
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

def csv_column_read(f, fieldnames, casts=None, start=None, end=None, reset=False):
	if casts is None or len(fieldnames) != len(casts):
		casts = [object] * len(fieldnames)

	def parse_row(row):
		row = {k: v for k, v in row.items() if k in fieldnames}
		return [row[key] for key in sorted(row.keys(), key=lambda k: fieldnames.index(k))]

	if reset:
		pos = f.tell()

	reader = csv.DictReader(f)
	rows = [[cast(field) for cast, field in zip(casts, fields)] for fields in (parse_row(row) for row in reader)]

	if reset:
		f.seek(pos)

	return [numpy.array(col[start:end], dtype=cast) for cast, col in zip(casts, zip(*rows))]

def csv_column_write(f, cols, fieldnames):
	writer = csv.DictWriter(f, fieldnames=fieldnames)
	writer.writeheader()

	for fields in zip(*cols):
		writer.writerow({fieldnames[i]: field for i, field in enumerate(fields)})

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
#         i \frac{\beta_4}{24} \frac{\partial^4 A}{\partial t^4}
#
# You apply the standard assumption, that A is of the form:
#
#                       A(z,t) = e^{i\Gamma z}a(t)
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
def init(t, eta, theta, phi):
	return [eta * math.exp(GAMMA * t) * (((GAMMA * (1+1j)) ** 0) * math.cos(theta) * cmath.exp(1j * GAMMA * t) + ((GAMMA * (1-1j)) ** 0) * math.sin(theta) * cmath.exp(1j * (phi - GAMMA * t))),
	        eta * math.exp(GAMMA * t) * (((GAMMA * (1+1j)) ** 1) * math.cos(theta) * cmath.exp(1j * GAMMA * t) + ((GAMMA * (1-1j)) ** 1) * math.sin(theta) * cmath.exp(1j * (phi - GAMMA * t))),
	        eta * math.exp(GAMMA * t) * (((GAMMA * (1+1j)) ** 2) * math.cos(theta) * cmath.exp(1j * GAMMA * t) + ((GAMMA * (1-1j)) ** 2) * math.sin(theta) * cmath.exp(1j * (phi - GAMMA * t))),
	        eta * math.exp(GAMMA * t) * (((GAMMA * (1+1j)) ** 3) * math.cos(theta) * cmath.exp(1j * GAMMA * t) + ((GAMMA * (1-1j)) ** 3) * math.sin(theta) * cmath.exp(1j * (phi - GAMMA * t)))]

def middle(eta, theta, phi):
	return math.log((2 * GAMMA**2) / (eta * (1 + math.sin(theta) ** 2))) / GAMMA
	# return math.log((2 * GAMMA**2) / eta) / GAMMA

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
	# return [b, c, d, -4*(GAMMA**4)*a]
	return [b, c, d, -4*(GAMMA**4)*a + a*abs(a)**2]

# Solver.
def solve(t0, t1, dt, start):
	solver = scipy.integrate.ode(soliton)
	solver.set_integrator("zvode", method="adams", order=12)
	solver.set_initial_value(start, t0)

	As = []
	while solver.successful() and solver.t < t1:
		solver.integrate(solver.t + dt)
		As.append((solver.t, solver.y[0]))

	ts, As = numpy.array(As).T
	return ts, As

def main():
	dt = 0.001

	# fig = plt.figure(figsize=(10, 10), dpi=80)
	# ax1 = latexify(fig.add_subplot("211"))
	# ax2 = latexify(fig.add_subplot("212"))

	num = 200
	thetaspace = numpy.linspace(0, 2*math.pi, num=num)
	phispace = numpy.linspace(0, math.pi, num=num)
	etaspace = 1e-14 * numpy.exp(numpy.linspace(0, 2*math.pi, num=num))

	fname = "interesting-%s.csv" % (random.randint(0, 99999999),)
	print(fname)
	with open(fname, "w") as f:
		writer = csv.DictWriter(f, fieldnames=["theta", "phi", "eta"])
		writer.writeheader()
		f.flush()

		# TODO: Vectorise
		interesting = []
		for theta in thetaspace:
			for phi in phispace:
				for eta in etaspace:
					t0 = 0
					t1 = 2*middle(eta, theta, phi)

					start = init(t0, eta, theta, phi)
					# print(".")
					ts, As = solve(t0, t1, dt, start)
					# ax1.plot(ts, numpy.vectorize(abs)(As))
					# ax2.plot(ts, numpy.vectorize(cmath.phase)(As))

					if ts.max() > 40:
						writer.writerow({"theta": theta, "phi": phi, "eta": eta})
						f.flush()

	theta = numpy.random.choice(thetaspace)
	phi = numpy.random.choice(phispace)
	eta = numpy.random.choice(etaspace)

	# theta=0.778564516011
	# eta=1.71455095837e-14
	# phi=0.0980274935414

	# theta = 3.94812014051
	# eta = 2.43519024939e-12
	# phi = 0.465630594321

	t0 = 0
	mid = middle(eta, theta, phi)
	t1 = 2*mid

	# start = init(t0, eta, theta, phi)
	# ts, As = solve(t0, t1, dt, start)

	# # now plot theoretical
	# ax1.set_title(r"{$\theta = %s, \eta = %s, \phi = %s$}" % (theta, eta, phi))

	# # plot integration
	# ax1.plot(ts, numpy.vectorize(abs)(As), 'k')
	# ax2.plot(ts, numpy.vectorize(cmath.phase)(As), 'k')

	# As = eta * numpy.exp(GAMMA * ts) * (((GAMMA * (1+1j)) ** 0) * numpy.cos(theta) * numpy.exp(1j * GAMMA * ts) + ((GAMMA * (1-1j)) ** 0) * numpy.sin(theta) * numpy.exp(1j * (phi - GAMMA * ts)))
	# # As = numpy.cos(theta) * numpy.exp(1j * ts) + numpy.sin(theta) * numpy.exp(1j * (phi - ts))
	# ax1.plot(ts, numpy.vectorize(abs)(As), 'r')
	# ax2.plot(ts, numpy.vectorize(cmath.phase)(As), 'r')

	# ax1.set_yscale("log")
	# ax1.set_axisbelow(True)
	# ax1.set_xlabel(r"Time ($\tau$)")
	# ax1.set_ylabel(r"Amplitude ($|A|$)")
	# ax1.set_xlim([t0, t1])

	# ax1.xaxis.set_ticks([mid], minor=True)
	# ax1.xaxis.grid(True, which="minor", color="k", linestyle=":")
	# ax2.xaxis.set_ticks([mid], minor=True)
	# ax2.xaxis.grid(True, which="minor", color="k", linestyle=":")

	# ax2.set_axisbelow(True)
	# ax2.set_xlabel(r"Time ($\tau$)")
	# ax2.set_ylabel(r"Phase")
	# ax2.set_xlim([t0, t1])
	# ax2.set_ylim([-math.pi, math.pi])

	# plt.legend()
	# fig.tight_layout()

	# plt.show()
	# plt.savefig(out)
	# print(out)


if __name__ == "__main__":
	main()
