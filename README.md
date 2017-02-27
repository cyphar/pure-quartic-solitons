## Pure Quartic Solitons ##
[![DOI](https://zenodo.org/badge/65123041.svg)](https://zenodo.org/badge/latestdoi/65123041)

```
Supervisors: Martijn de Sterke, Andrea Blanco Redondo
Institution: University of Sydney
```

A research project done as part of the Talented Student Program at the
University of Sydney. [This paper][pqs-paper] is the core thing this
project is planning to explore.

### Goal ###

The goal of this project is to produce a numerical, soliton-like
solution for the differential equation:

```latex
\frac{\partial A}{\partial z} =
		i \frac{\beta_4}{24} \frac{\partial^4 A}{\partial t^4} +
		i \gamma_{eff} {\left|A\right|}^{2} A
```

This is much harder than it sounds, since it essentially resolves to
being a boundary value problem with 8 real parameters (4 complex
parameters), but it can be reduced to 3 real parameters through a bunch
of clever tricks.  Similarly, the search can be constrained using
further tricks. And finally, it all boils down to a bruteforce of the
search space.

### License ###

This project is licensed under the GNU General Public License, version 3 or
later.

```
Copyright (C) 2016 Aleksa Sarai

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
```

[pqs-paper]: http://arxiv.org/abs/1508.03120
