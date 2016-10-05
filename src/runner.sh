#!/bin/zsh
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

usage() {
	echo "usage: $0 -p NUM_PARTITIONS -- [arguments]" >&2
	exit 1
}

# Get -p flag.
export NUM_PARTITIONS=0
export OUTDIR="."
while getopts ":p:o:" o; do
	case "${o}" in
		p)
			NUM_PARTITIONS="${OPTARG}"
			;;
		o)
			OUTDIR="${OPTARG}"
			;;
		*)
			usage
			;;
	esac
done

# Ensure NUM_PARTITIONS > 0.
if [ "$NUM_PARTITIONS" -le 0 ]; then
	usage
fi

# Get other arguments.
shift ${OPTIND}-1


mkdir -p "${OUTDIR}"
PREFIX="${OUTDIR}/results_$RANDOM"
echo "OUTPUT :: $PREFIX.index.csv"

# Thanks GNU parallel!
(
	for i in {1..${NUM_PARTITIONS}}; do
		echo $((${i}-1))
	done
) | parallel ./src/searcher.py --num-partitions ${NUM_PARTITIONS} -i {} -o ${PREFIX}.{}.csv $*
