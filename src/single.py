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

import os
import sys
import csv
import cmath
import math
import argparse
import random

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid.anchored_artists import AnchoredText
import numpy
import numpy.random
import scipy
import scipy.integrate

matplotlib.rcParams.update({
	"backend": "ps",
	"text.usetex": True,
	"text.latex.preamble": [r"\usepackage[dvips]{graphicx}"],
	"axes.labelsize": 13, # fontsize for x and y labels (was 10)
	"axes.titlesize": 13,
	"font.size": 13, # was 10
	"legend.fontsize": 13, # was 10
	"xtick.labelsize": 13,
	"ytick.labelsize": 13,
	"font.family": "serif", # ???
})

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

def csv_column_write(f, cols, fieldnames):
	import csv

	writer = csv.DictWriter(f, fieldnames=fieldnames)
	writer.writeheader()

	for fields in zip(*cols):
		writer.writerow({fieldnames[i]: field for i, field in enumerate(fields)})

def csv_column_read(f, fieldnames, casts=None, start=None, end=None, reset=False):
	if casts is None or len(fieldnames) != len(casts):
		casts = [object] * len(fieldnames)

	def parse_row(row):
		row = {k: v for k, v in row.items() if k in fieldnames}
		return [row[key] for key in sorted(row.keys(), key=lambda k: fieldnames.index(k))]

	if reset:
		pos = f.tell()

	reader = csv.DictReader(f)
	rows = [[cast(field) for cast, field in zip(casts, fields)] for fields in (parse_row(row) for row in reader if row)]

	if reset:
		f.seek(pos)

	return [numpy.array(col[start:end], dtype=cast) for cast, col in zip(casts, zip(*rows))]

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
	alpha = eta * math.exp(GAMMA * t) * (((GAMMA * (1+1j)) ** 0) * math.cos(theta) * cmath.exp(1j * GAMMA * t) + ((GAMMA * (1-1j)) ** 0) * math.sin(theta) * cmath.exp(1j * (phi - GAMMA * t)))
	beta  = eta * math.exp(GAMMA * t) * (((GAMMA * (1+1j)) ** 1) * math.cos(theta) * cmath.exp(1j * GAMMA * t) + ((GAMMA * (1-1j)) ** 1) * math.sin(theta) * cmath.exp(1j * (phi - GAMMA * t)))
	gamma = eta * math.exp(GAMMA * t) * (((GAMMA * (1+1j)) ** 2) * math.cos(theta) * cmath.exp(1j * GAMMA * t) + ((GAMMA * (1-1j)) ** 2) * math.sin(theta) * cmath.exp(1j * (phi - GAMMA * t)))
	delta = eta * math.exp(GAMMA * t) * (((GAMMA * (1+1j)) ** 3) * math.cos(theta) * cmath.exp(1j * GAMMA * t) + ((GAMMA * (1-1j)) ** 3) * math.sin(theta) * cmath.exp(1j * (phi - GAMMA * t)))

	# Real then imaginary.
	return [alpha.real, beta.real, gamma.real, delta.real,
			alpha.imag, beta.imag, gamma.imag, delta.imag]

def middle(eta, theta, phi):
	return math.log((2 * GAMMA**2) / (eta * (1 + math.sin(theta) ** 2))) / GAMMA

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
	aR, bR, cR, dR, aI, bI, cI, dI = y
	return [bR, cR, dR, -4*(GAMMA**4)*aR + aR * (aR**2 + aI**2),
			bI, cI, dI, -4*(GAMMA**4)*aI + aI * (aR**2 + aI**2)]

# Solver.
def solve(t0, t1, dt, start):
	solver = scipy.integrate.ode(soliton)
	solver.set_integrator("dop853", nsteps=2000)
	# solver.set_integrator("vode", method="adams", nsteps=2000, max_step=dt)
	solver.set_initial_value(start, t0)

	As = []
	while solver.successful() and solver.t < t1:
		solver.integrate(solver.t + dt)
		As.append((solver.t, solver.y[0] + 1j*solver.y[4]))

	ts, As = numpy.array(As).T
	return ts, As

def main(config):
	eta = config.eta
	theta = config.theta
	phi = config.phi

	fig = plt.figure(figsize=(5, 5), dpi=80)
	ax1 = latexify(fig.add_subplot("%d11" % (1+config.show_phase,)))
	if config.show_phase:
		ax2 = latexify(fig.add_subplot("212"))
	else:
		ax2 = None

	t0 = 0
	mid = middle(eta, theta, phi)
	t1 = 2*mid

	start = init(t0, eta, theta, phi)
	ts, As = solve(t0, t1, config.dt, start)

	# now plot theoretical
	# ax1.set_title(r"{$\theta = %s, \eta = %s, \phi = %s$}" % (theta, eta, phi))

	# plot integration
	ax1.plot(ts, numpy.vectorize(abs)(As), 'k')
	if ax2 is not None:
		ax2.plot(ts, numpy.vectorize(cmath.phase)(As), 'k')

	# plot first order approximation.
	if config.first_order:
		n1 = GAMMA * numpy.cos(theta)
		n2 = GAMMA * numpy.sin(theta) * numpy.exp(1j * phi)
		linAs = eta * numpy.exp(GAMMA * ts) * (n1 * numpy.exp(1j * GAMMA * ts) + n2 * numpy.exp(-1j * GAMMA * ts))
		ax1.plot(ts, numpy.vectorize(abs)(linAs), 'r--')
		if ax2 is not None:
			ax2.plot(ts, numpy.vectorize(cmath.phase)(linAs), 'r--')

		if config.fill_area:
			Is = numpy.vectorize(abs)(As)
			linIs = numpy.vectorize(abs)(linAs)

			# Cut down the time domain so we don't take into account anything before the estimated peak.
			filt = ts >= mid
			ts = ts[filt]
			Is = Is[filt]
			linIs = linIs[filt]

			# Remove entries once the non-linear terms become out-of-hand.
			grad = numpy.gradient(Is)
			filt = grad < 100
			ts = ts[filt]
			Is = Is[filt]
			linIs = linIs[filt]

			# We need to ignore parts when Is > linIs.
			filt = Is <= linIs
			ts = ts[filt]
			Is = Is[filt]
			linIs = linIs[filt]

			# Fill.
			ax1.fill_between(ts, Is, linIs, color='orange')

	ax1.set_yscale(config.scale)
	ax1.set_axisbelow(True)
	ax1.set_xlabel(r"Time ($\tau$)")
	ax1.set_ylabel(r"Amplitude ($|b|$)")
	ax1.set_xlim([t0, t1])

	if config.scale == "log":
		ax1.set_ylim([None, 1e11])
	elif config.scale == "linear":
		ax1.set_ylim([0, 3])

	# ax1.xaxis.set_ticks([mid], minor=True)
	# ax1.xaxis.grid(True, which="minor", color="k", linestyle=":")
	# if ax2 is not None:
		# ax2.xaxis.set_ticks([mid], minor=True)
		# ax2.xaxis.grid(True, which="minor", color="k", linestyle=":")

	if ax2 is not None:
		ax2.set_axisbelow(True)
		ax2.set_xlabel(r"Time ($\tau$)")
		ax2.set_ylabel(r"Phase")
		ax2.set_xlim([t0, t1])
		ax2.set_ylim([-math.pi, math.pi])

		# Set ticks to represent angle.
		ticks = numpy.linspace(-1, 1, num=9)
		ax2.set_yticks([x*math.pi for x in ticks])
		ax2.yaxis.set_major_formatter(matplotlib.ticker.FixedFormatter([r"$%s \pi$" % (x,) for x in ticks]))

	plt.legend()
	fig.tight_layout()

	if config.out:
		plt.savefig(config.out)
	else:
		plt.show()


if __name__ == "__main__":
	def __wrapped_main__():
		parser = argparse.ArgumentParser(description="Searches the parameter space (eta, theta, phi) for the PQS ODE using the set of metrics given.")
		# save plot
		parser.add_argument("--out", dest="out", type=str, default=None, help="Output to given filename rather than displaying in an interactive window (default: disabled)")
		# integrator arguments
		parser.add_argument("-dt", dest="dt", type=float, default=0.01, help="Time spacing for integration (default: 0.01).")
		# plot arguments
		parser.add_argument("-fo", "--first-order", dest="first_order", default=False, action="store_const", const=True, help="Plot the first order approximation of the curve.")
		parser.add_argument("--no-first-order", dest="first_order", default=False, action="store_const", const=False, help="Do not plot the first order approximation of the curve (default).")
		parser.add_argument("-fa", "--fill-area", dest="fill_area", default=False, action="store_const", const=True, help="Fill the area between the first order approximation and the integral.")
		parser.add_argument("--no-fill-area", dest="fill_area", default=False, action="store_const", const=False, help="Do not fill the area between the first order approximation and the integral (default).")
		parser.add_argument("-pp", "--plot-phase", dest="show_phase", default=False, action="store_const", const=True, help="Show the phase of the curve.")
		parser.add_argument("--no-plot-phase", dest="show_phase", default=False, action="store_const", const=False, help="Do not show the phase of the curve. (default)")
		parser.add_argument("-sc", "--scale",  dest="scale", default="log", help="The scaling to use for the y axis (default: log).")
		# parameter arguments
		parser.add_argument("-e", "--eta", dest="eta", type=float, required=True, help="Eta value.")
		parser.add_argument("-t", "--theta", dest="theta", type=float, required=True, help="Theta value.")
		parser.add_argument("-p", "--phi", dest="phi", type=float, required=True, help="Phi value.")

		config = parser.parse_args()

		if config.fill_area and not config.first_order:
			print("--fill-area requires --first-order")
			sys.exit(1)

		main(config)

	__wrapped_main__()
