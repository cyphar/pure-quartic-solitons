#!/usr/bin/env python3
# Copyright (C) 2015 Aleksa Sarai <cyphar@cyphar.com>
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
import csv
import sys
import math
import numpy
import argparse

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.colors
import matplotlib.ticker
import matplotlib.pyplot as plt
import cmocean

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

def positions(ndarray):
	return zip(*numpy.where(numpy.ones_like(ndarray)))

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

# Reads in the csv file for the shoebox and outputs the relevant arrays.
def read_shoebox(fname, metric):
	with open(fname) as f:
		return csv_column_read(f, ["eta", "theta", metric], casts=[float, float, float])

def plot_shoebox(config, ax, fname, metric="metric"):
	# Get the columns.
	etas, thetas, metrics = read_shoebox(fname, metric=metric)

	# We need to figure out what the eta and theta ranges are.
	r_etas = numpy.unique(etas)
	r_thetas = numpy.unique(thetas)

	# Set up the X-Y matrix (eta, theta) for the plot.
	y_idx, x_idx = numpy.mgrid[0:len(r_etas), 0:len(r_thetas)]

	# Convert from idx to the actual X-Y value.
	y = r_etas[y_idx]
	x = r_thetas[x_idx]

	# Get the values.
	z = numpy.zeros_like(x)
	for idx, eta in enumerate(r_etas):
		try:
			z[idx,...] = metrics[etas == eta]
		except ValueError:
			continue

	# Use phase for phase "metric".
	if "phi" in metric:
		cmap = cmocean.cm.phase
	else:
		cmap = "viridis"
		# z[z > 0] = numpy.log(z[z > 0])

	# Plot all the things.
	pcm = ax.pcolor(x, y, z, cmap=cmap, vmin=z.min(), vmax=z.max())
	if config.angle_ticks:
		ticks = numpy.linspace(0, 1, num=5)
		ax.set_xticks([x*math.pi for x in ticks])
		ax.xaxis.set_major_formatter(matplotlib.ticker.FixedFormatter([r"$%s \pi$" % (x,) for x in ticks]))

	ax.set_xlim(eval(config.theta_space))
	ax.set_ylim(eval(config.eta_space))

	# ax.set_title("Shoebox [metric=%s]" % (metric,))

	return pcm

def main(config):
	print("PLOT :: %s [metric=%s]" % (config.file, config.metric))

	size = len(config.theta_space)
	theta_space = config.theta_space
	config.eta_space = config.eta_space[0]

	f, plots = plt.subplots(1 + config.best_phi, size, figsize=(5, 5), dpi=80, sharex='col', sharey='row')

	if numpy.array([plots]).shape == (1,):
		plots = numpy.array([plots])

	if len(plots.shape) == 1:
		plots = numpy.array([plots]).T

	for idx, axes in enumerate(plots.T):
		ax1 = ax2 = None
		if len(axes) >= 1:
			ax1 = latexify(axes[0])
		if len(axes) >= 2:
			ax2 = latexify(axes[1])

		config.theta_space = theta_space[idx]
		print(config.theta_space)
		if ax1 is not None:
			pcm1 = plot_shoebox(config, ax1, config.file[0], metric=config.metric)
		if ax2 is not None:
			pcm2 = plot_shoebox(config, ax2, config.file[0], metric=config.metric+"_phi")

		if ax1 is not None:
			f.colorbar(pcm1, ax=ax1)
		if ax2 is not None:
			if config.angle_ticks:
				ticks = numpy.linspace(0, 2, num=9)
				cbar = f.colorbar(pcm2, ax=ax2, ticks=[x*math.pi for x in ticks])
				cbar.ax.set_yticklabels([r"$%s \pi$" % (x,) for x in ticks])
			else:
				cbar = f.colorbar(pcm2, ax=ax2)

		if ax1 is not None:
			ax1.set_ylabel(r"$\eta$")
			if ax1 is None:
				ax1.set_xlabel(r"$\theta$")
		if ax2 is not None:
			ax2.set_xlabel(r"$\theta$")

	# plt.legend()
	f.subplots_adjust(hspace=-0.2)
	f.tight_layout()

	if config.out:
		plt.savefig(config.out)
	else:
		plt.show()

if __name__ == "__main__":
	def __wrapped_main__():
		parser = argparse.ArgumentParser(description="Plots the shape of the parameter space (eta, theta, phi) with the colour being set by the given metric value.")
		# save argument
		parser.add_argument("--out", dest="out", type=str, default=None, help="Output to given filename rather than displaying in an interactive window (default: disabled)")
		# metric arguments
		parser.add_argument("-m", "--metric", dest="metric", type=str, default="depth", help="Metric to plot from {depth, linear} (default: [depth]).")
		# plotting arguments
		parser.add_argument("-sp", "--show-best-phi", dest="best_phi", action="store_const", const=True, default=False, help="Show best phi plot.")
		parser.add_argument("--no-show-best-phi", dest="best_phi", action="store_const", const=False, default=False, help="Do not show best phi plot. (default)")
		parser.add_argument("-a", "--angle-ticks", dest="angle_ticks", action="store_const", const=True, default=True, help="Use angle ticks (have multiples of pi when plotting angles) (default).")
		parser.add_argument("--no-angle-ticks", dest="angle_ticks", action="store_const", const=False, default=True, help="Do not use angle ticks.")
		# shoebox arguments
		parser.add_argument("-eS", "--eta-space", dest="eta_space", action="append", default=[], help="Limits of eta-space shoebox. (default: (1e-10,1e-10*math.exp(math.pi))).")
		parser.add_argument("-tS", "--theta-space", dest="theta_space", action="append", default=[], help="Limits of theta-space shoebox. (default: (0,math.pi)).")
		parser.add_argument("file", nargs=1)

		config = parser.parse_args()
		if not config.eta_space:
			config.eta_space = ["(1e-10,1e-10*math.exp(math.pi))"]
		if not config.theta_space:
			config.theta_space = ["(0,math.pi)"]

		if len(config.eta_space) > 1:
			raise NotImplementedError("eta_space multi-range not implemented")

		main(config)

	__wrapped_main__()
