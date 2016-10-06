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
	echo "usage: $0 <file>..." >&2
	exit 1
}

[[ "$#" -eq 0 ]] && usage

# Print first line from first file, then sort the rest.
head -n1 "$1"
(
	for file in $*; do
		tail -n+2 "$file"
	done
) | sort -g
