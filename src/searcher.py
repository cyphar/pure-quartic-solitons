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

import numpy
import numpy.random
import scipy
import scipy.integrate

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
	return math.log((2*GAMMA**2) / (eta * (1 + math.sin(theta) ** 2))) / GAMMA

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

def metric_depth(ts, As, eta, theta, phi):
	mid = middle(eta, theta, phi)
	Is = numpy.vectorize(abs)(As)

	# Cut down time domain so that we don't detect minima before the estimate.
	filt = ts >= mid
	ts = ts[filt]
	Is = Is[filt]

	# Get lowest dip value.
	i_dip = numpy.argmin(Is)
	t_dip = ts[i_dip]
	I_dip = Is[i_dip]

	# Get highest peak value that happened after mid but before the dip.
	filt = ts < t_dip
	# ts = ts[filt]
	Is = Is[filt]

	if not Is.shape[0]:
		return None

	# TODO: Handle cases where the maximum is before the peak.
	I_peak = numpy.amax(Is)

	return (I_peak - I_dip) / I_peak

def metric_linear(ts, As, eta, theta, phi):
	mid = middle(eta, theta, phi)
	Is = numpy.vectorize(abs)(As)

	# First-order linear approximation.
	n1 = GAMMA * numpy.cos(theta)
	n2 = GAMMA * numpy.sin(theta) * numpy.exp(1j * phi)
	linAs = eta * numpy.exp(GAMMA * ts) * (n1 * numpy.exp(1j * GAMMA * ts) + n2 * numpy.exp(-1j * GAMMA * ts))
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

	# We'll operate on the log values for sanity reasons.
	Is = numpy.log(Is)
	linIs = numpy.log(linIs)

	# Get the difference between linIs and Is and compute a metric from that.
	diff = linIs - Is

	# We need to ignore parts when Is > linIs.
	filt = Is <= linIs
	ts = ts[filt]
	diff = diff[filt]
	return abs(numpy.trapz(diff, ts))
	# return numpy.sum(diff)

METRICS = {
	"depth": metric_depth,
	"linear": metric_linear,
}

def main(config):
	# Make sure we create the output directory first.
	if config.out_directory:
		try:
			os.mkdir(config.out_directory)
		except OSError:
			print("FAIL :: failed to create directory")

	# Convert *_space limits to something sane.
	etaspace = numpy.array([])
	for etas in config.eta_space:
		rng = eval(etas)
		try:
			high, low = rng
			space = numpy.linspace(low, high, num=config.eta_samples)
		except TypeError:
			space = numpy.array([rng])
		etaspace = numpy.concatenate((etaspace, space))
	thetaspace = numpy.array([])
	for thetas in config.theta_space:
		rng = eval(thetas)
		try:
			high, low = rng
			space = numpy.linspace(low, high, num=config.theta_samples)
		except TypeError:
			space = numpy.array([rng])
		thetaspace = numpy.concatenate((thetaspace, space))
	phispace = numpy.array([])
	for phis in config.phi_space:
		rng = eval(phis)
		try:
			high, low = rng
			space = numpy.linspace(low, high, num=config.phi_samples)
		except TypeError:
			space = numpy.array([rng])
		phispace = numpy.concatenate((phispace, space))

	print("ETA   :: %s <- %s" % (config.eta_space, config.eta_samples))
	print("THETA :: %s <- %s" % (config.theta_space, config.theta_samples))
	print("PHI   :: %s <- %s" % (config.phi_space, config.phi_samples))

	# Partition space according to the arguments.Samples. Only partition one of
	# these dimensions, for obvious reasons.
	etaspace = numpy.array_split(etaspace, config.partitions)[config.index]
	# thetaspace = numpy.array_split(thetaspace, config.partitions)[config.index]
	# phispace = numpy.array_split(phispace, config.partitions)[config.index]

	# Nothing is assigned to us (shame).
	if not etaspace.shape[0]:
		return

	# Result output.
	if config.out_results:
		out = config.out_results
	else:
		out = "results-%d.csv" % (random.randint(0, 999999),)
	print("OUTPUT :: %s" % (out,))

	partial = set()
	if os.path.isfile(out):
		with open(out, "r") as resf:
			# First figure out the metrics.
			header = resf.readline().strip()
			new_metrics = [m[:-len("_phi")] for m in header.split(',') if m.endswith("_phi")]

			# Warn the user.
			if new_metrics != config.metrics:
				print("METRICS :: updated to match old file")
			config.metrics = new_metrics

			# Reset.
			resf.seek(0)

			# Get what has already been computed. We use strings because
			# floating point comparison is bad.
			part_etas, part_thetas = csv_column_read(resf, ["eta", "theta"], [str, str])
			partial = {(eta, theta) for eta, theta in numpy.array([part_etas, part_thetas]).T}

	# Get the set of metrics requested.
	metrics = [METRICS[m] for m in config.metrics]
	print("METRICS :: %s (%d)" % (config.metrics, len(metrics)))

	# TODO: Vectorise
	with open(out, "a") as f:
		writer = csv.DictWriter(f, fieldnames=["eta", "theta"] + config.metrics + [m+"_phi" for m in config.metrics])

		# Don't write the header if we already have partial.
		if not partial:
			writer.writeheader()
		else:
			f.write('\n')
		f.flush()

		for eta in etaspace:
			for theta in thetaspace:
				# Skip if already existed in partial results file.
				if (str(eta), str(theta)) in partial:
					print("!! %s", (eta, theta))
					continue

				# Find the metrics for phi given (eta, theta).
				values = []
				for _ in enumerate(metrics):
					values.append([])

				# Search through phi.
				for phi in phispace:
					t0 = 0
					t1 = 2*middle(eta, theta, phi)

					path = ""
					if config.out_directory:
						path = "%s/eta:%s/theta:%s/phi:%s.csv" % (config.out_directory, eta, theta, phi)

					ts = As = None
					if path and os.path.isfile(path):
						sys.stdout.write("*")
						sys.stdout.flush()
						with open(path, "r") as intf:
							try:
								ts, As = csv_column_read(intf, ["t", "A"], [complex, complex])
							except ValueError:
								# There was some issue with the formatting of
								# the output -- just redo the integration.
								ts = As = None

					if (ts is None) or (As is None):
						start = init(t0, eta, theta, phi)
						sys.stdout.write(".")
						sys.stdout.flush()
						ts, As = solve(t0, t1, config.dt, start)

						# Save the intermediate integration steps.
						if config.out_directory:
							path = "%s/eta:%s/theta:%s/phi:%s.csv" % (config.out_directory, eta, theta, phi)
							try:
								os.makedirs(os.path.dirname(path))
							except FileExistsError:
								pass
							with open(path, "w") as intf:
								csv_column_write(intf, [ts, As], ["t", "A"])

					# Compute the metrics.
					for idx, m in enumerate(metrics):
						met = m(ts, As, eta, theta, phi)
						if met is not None:
							values[idx].append((met, phi))

				# Get best metric value for each metric.
				bests = [None] * len(metrics)
				for idx, v in enumerate(values):
					if v:
						v = numpy.array(v)
						v = v[numpy.lexsort((v[:,0],))]
					else:
						v = [(0, 0)]

					bests[idx] = v[-1]

				# Debug output.
				print('eta=%s,theta=%s => ' % (eta, theta), bests)

				# Generate the row.
				row = {"eta": eta, "theta": theta}
				for idx, m in enumerate(config.metrics):
					row[m] = bests[idx][0]
					row[m+"_phi"] = bests[idx][1]

				# Output and flush.
				writer.writerow(row)
				f.flush()


if __name__ == "__main__":
	def __wrapped_main__():
		parser = argparse.ArgumentParser(description="Searches the parameter space (eta, theta, phi) for the PQS ODE using the set of metrics given.")
		# integrator arguments
		parser.add_argument("-dt", dest="dt", type=float, default=0.01, help="Time spacing for integration (default: 0.01).")
		# metric arguments
		parser.add_argument("-m", "--metric", dest="metrics", action="append", default=["depth"], help="Set of metrics from {depth, linear} to compute (default: [depth]).")
		# multithreading arguments
		parser.add_argument("-p", "--num-partitions", dest="partitions", type=int, default=1, help="Number of partitions to split the space into (default: 1).")
		parser.add_argument("-i", "--partition-index", dest="index", type=int, default=0, help="The 0-indexed partition index to be searched (default: 0).")
		# output arguments
		parser.add_argument("-s", "--save-directory", dest="out_directory", type=str, default=None, help="Output directory for integration data (default: none).")
		parser.add_argument("-o", "--save-results", dest="out_results", type=str, default=None, help="Output path for search results (default: random).")
		# shoebox arguments
		parser.add_argument("-eS", "--eta-space", dest="eta_space", action="append", default=[], help="Limits of eta-space shoebox. (default: (1e-10,1e-10*math.exp(math.pi))).")
		parser.add_argument("-tS", "--theta-space", dest="theta_space", action="append", default=[], help="Limits of theta-space shoebox. (default: (0,math.pi)).")
		parser.add_argument("-pS", "--phi-space", dest="phi_space", action="append", default=[], help="Limits of phi-space shoebox. (default: (0,2*math.pi)).")
		# sampling arguments
		parser.add_argument("-es", "--eta-samples", dest="eta_samples", type=int, default=100, help="Number of samples taken in eta-space. (default: 100).")
		parser.add_argument("-ts", "--theta-samples", dest="theta_samples", type=int, default=100, help="Number of samples taken in theta-space. (default: 100).")
		parser.add_argument("-ps", "--phi-samples", dest="phi_samples", type=int, default=500, help="Number of samples taken in phi-space. (default: 500).")

		config = parser.parse_args()

		if not config.eta_space:
			config.eta_space = ["(1e-10,1e-10*math.exp(math.pi))"]
		if not config.theta_space:
			config.theta_space = ["(0,math.pi)"]
		if not config.phi_space:
			config.phi_space = ["(0,2*math.pi)"]

		main(config)

	__wrapped_main__()
