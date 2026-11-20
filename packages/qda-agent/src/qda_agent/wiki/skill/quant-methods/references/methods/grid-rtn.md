# grid-rtn
Addresses: a group's weights must map onto a fixed number of levels. The scheme alone (bit width, group size, zero point) sets the error of that mapping before any data-driven correction. Measured as: per group, the round-to-nearest error ‖W − Q(W)‖ under the min/max grid, and its spread across modules.
Mechanism: Each group takes its scale (and zero point, if asymmetric) from the weights alone, and every weight is rounded to the nearest code, with no calibration data and no error feedback. It is the data-free reference of a scheme; it shows whether a failed build is a method failure or a scheme limit.
Effect: every weight inside the range lands on its nearest level, so each weight's error is at most half the step range/(2^b − 1); no calibration statistic enters.
Preconditions: none beyond the scheme being servable (the kernel's accepted bit widths, group sizes and zero-point support; see grid-scheme).
Inference overhead: none beyond the scheme itself.
Pitfalls: the round-to-nearest reference is valid only for the same scheme; a reference built with a different observer, group size or zero-point convention is a different format and not a reference for that scheme. Second: a module that stayed in the source dtype makes the reference error look smaller than the scheme allows; the coverage report of the build shows it.
Source: no single publication (standard round-to-nearest baseline).
