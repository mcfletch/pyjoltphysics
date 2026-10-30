// The module's binding functions, one per area, called from NB_MODULE.
#pragma once

#include <nanobind/nanobind.h>

namespace pj {

void bind_world(nanobind::module_ &m);

}  // namespace pj
