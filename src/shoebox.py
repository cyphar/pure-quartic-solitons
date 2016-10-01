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

import matplotlib
import matplotlib.colors
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

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

def plot_shoebox(ax, fname, metric="metric"):
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

	# Plot all the things.
	# ax.pcolor(x, y, z, cmap='viridis', vmin=z.min(), vmax=z.max())
	pcm = ax.pcolor(x, y, z, cmap='PuBu_r', vmin=z.min(), vmax=z.max())
	ax.set_xlim([x.min(), x.max()])
	ax.set_xlabel(r"$\theta$")
	ax.set_ylim([y.max(), y.min()])
	ax.set_ylabel(r"$\eta$")

	ax.set_title("Shoebox [metric=%s]" % (metric,))

	return pcm

def main():
	if len(sys.argv) != 2:
		print("usage: shoebox.py <file.csv>")
		os.exit(1)

	print("plotting %s!" % (sys.argv[1]))

	fig = plt.figure(figsize=(10, 10), dpi=80)
	ax1 = latexify(fig.add_subplot("211"))
	ax2 = latexify(fig.add_subplot("212"))

	pcm1 = plot_shoebox(ax1, sys.argv[1], metric="metric")
	pcm2 = plot_shoebox(ax2, sys.argv[1], metric="phi")

	fig.colorbar(pcm1, ax=ax1)
	fig.colorbar(pcm2, ax=ax2)

	# plt.legend()
	fig.tight_layout()
	plt.show()

main()
